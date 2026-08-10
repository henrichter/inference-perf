# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conversation replay data generator for agentic workload benchmarking.

Closed-loop continuous replenishment model
------------------------------------------
Each slot (preferred_worker_id) maps to one concurrent conversation. When a
conversation completes all its turns, the slot immediately resets to a fresh
LocalUserSession and starts a new conversation from turn 0. This models
steady-state production traffic where a new conversation begins as soon as
the previous one ends.

At steady state, the C active conversations are uniformly distributed across
turn 0..N, so the mean KV-cache context across all active slots matches the
production mean across all active slots.

Usage
-----
Set ``num_conversations = C`` (the concurrency level) so each slot owns
exactly one conversation. Set ``num_requests = C × turns × num_rounds`` to
run ``num_rounds`` complete conversations per slot. The first round is
warmup; report throughput from round 2 onward.

Run each concurrency level as a separate benchmark (fresh state, not stages
of one run) to avoid context accumulating across concurrency levels.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from multiprocessing.managers import SyncManager
from typing import Any, Dict, Generator, List, Optional

import numpy as np

from aiohttp import ClientResponse
from inference_perf.apis.base import InferenceAPIData, InferenceInfo, LazyLoadInferenceAPIData
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.config import (
    APIConfig,
    APIType,
    ConversationReplayConfig,
    DataConfig,
    Distribution,
)
from inference_perf.datagen.replay_graph_session_datagen import (
    ReplayGraphSessionGeneratorBase,
    ReplaySession,
)
from inference_perf.datagen.replay_graph_types import (
    GraphCall,
    GraphEvent,
    InputSegment,
    ReplayGraph,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.numeric.distribution import sample_from_distribution

from .base import DataGenerator, LazyLoadDataMixin

logger = logging.getLogger(__name__)


class _ConversationReplayAPIData(UserSessionCompletionAPIData):
    """UserSessionCompletionAPIData with tool call latency simulation.

    After the model generates a response, sleeps for ``tool_call_latency_sec``
    seconds *before* releasing the session lock. This correctly serialises:

        model inference → sleep(tool_call_latency) → next model inference

    while allowing the GPU to serve other concurrent conversations during the
    sleep (the asyncio event loop is not blocked; other slots' requests run).

    If ``tool_call_latency_sec == 0`` the behaviour is identical to the
    parent class.
    """

    tool_call_latency_sec: float = 0.0

    async def process_response(
        self,
        response: ClientResponse,
        config: APIConfig,
        tokenizer: CustomTokenizer,
        lora_adapter: Optional[str] = None,
    ) -> InferenceInfo:
        # Run the base completion response handler (sets self.model_response,
        # records timing metrics) WITHOUT yet releasing the session lock.
        # We call CompletionAPIData directly to skip UserSessionCompletionAPIData's
        # update_context call so we can inject the sleep in between.
        inference_info = await CompletionAPIData.process_response(self, response, config, tokenizer, lora_adapter)
        self.update_inference_info(inference_info)

        # Simulate tool execution latency while holding the session lock.
        # The next turn of this conversation cannot start until the sleep
        # completes; other conversations' turns run freely during the wait.
        if self.tool_call_latency_sec > 0:
            await asyncio.sleep(self.tool_call_latency_sec)

        # Release the session lock by updating context (allows next turn).
        self.user_session.update_context(self.prompt + " " + self.model_response)
        return inference_info

    async def process_failure(
        self,
        response: Optional[ClientResponse],
        config: APIConfig,
        tokenizer: CustomTokenizer,
        exception: Exception,
        lora_adapter: Optional[str] = None,
    ) -> Optional[InferenceInfo]:
        # On failure, release the lock without sleeping (no tool was called).
        inference_info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0)))
        self.update_inference_info(inference_info)
        self.user_session.update_context(self._session_context)
        return inference_info


@dataclass
class ConversationBlueprint:
    """Pre-computed plan for a single conversation."""

    conversation_id: int
    num_turns: int
    system_prompt: str
    system_prompt_tokens: int
    dynamic_system_prompt: str = ""
    turn_prompts: List[str] = field(default_factory=list)
    turn_input_lens: List[int] = field(default_factory=list)
    turn_output_lens: List[int] = field(default_factory=list)
    turn_tool_call_latencies: List[float] = field(default_factory=list)
    # Per-turn cumulative input token count (system + all prior turns + this user
    # turn), resolved at build time from the ignore_eos-pinned lengths. Lets
    # _blueprint_to_graph assemble the growing-prefix graph with no tokenizer calls.
    turn_total_input_tokens: List[int] = field(default_factory=list)


class _ConversationBlueprintBuilder:
    """Shared synthetic-conversation blueprint generation.

    Owns the deterministic content generation (system prompts, per-turn prompts,
    output lengths, tool-call latencies) common to the closed-loop
    ``ConversationReplayDataGenerator`` (a ``DataGenerator``) and the open-loop
    ``ConversationSessionGenerator`` (a ``SessionGenerator``). Subclasses call
    ``_init_generation`` from ``__init__`` after ``super().__init__`` has set
    ``self.tokenizer``, then ``_build_conversations``. Nothing here has any
    dispatch- or session-lifecycle dependency, so both generators build identical
    blueprints from the same config + seed.
    """

    tokenizer: Optional[CustomTokenizer]

    # Headroom reserved below max_model_len for a single request (input + output),
    # matching the closed-loop LocalUserSession path (see user_session.py:148) so both
    # generators agree on what "fits" the backend context window.
    _CONTEXT_SAFETY_BUFFER = 200

    @property
    def _context_budget(self) -> int:
        """Total tokens (input + output) a single request may occupy."""
        return self.max_model_len - self._CONTEXT_SAFETY_BUFFER

    def _plan_within_budget(
        self, sys_tokens: int, input_lens: List[int], output_lens: List[int]
    ) -> tuple[int, List[int], List[int], List[int]]:
        """Bound a conversation to the context window entirely by arithmetic.

        Because the run uses ``ignore_eos``, each turn's output length is pinned to
        ``output_lens[k]``, so the whole growing prefix is known up front:

            total_input(0) = eff_sys + u_0
            total_input(k) = total_input(k-1) + eff_out(k-1) + u_k   (k > 0)

        and every request must satisfy ``total_input(k) + eff_out(k) <= budget``.
        We resolve the *effective* lengths here, before any text is generated, so
        content that would not fit is never sampled or materialised. Two clamps and
        one stop (mirroring the agreed overflow model):

        * Turn 0 system clamp: ``eff_sys = min(sys, budget - u_0 - out_0)`` (floor 0);
          if ``u_0 + out_0`` alone still overflows, clamp ``out_0`` too.
        * Turn k>0 input-overflow: if ``total_input(k) >= budget`` the prefix leaves
          no room for output -> stop; the conversation ends at turn k-1.
        * Turn k output-overflow: if ``total_input(k) + out_k > budget`` clamp
          ``out_k = budget - total_input(k)`` and keep the turn.

        Returns ``(eff_sys_tokens, eff_input_lens, eff_output_lens, per_turn_total_input)``
        where the three per-turn lists are truncated to the effective turn count.
        """
        budget = self._context_budget
        n = len(input_lens)
        eff_inputs: List[int] = []
        eff_outputs: List[int] = []
        totals: List[int] = []

        # Turn 0 must fit; the user turn is the genuinely new content and is kept, so
        # the system prompt absorbs the clamp first, then out_0 if still needed.
        u0 = input_lens[0] if n > 0 else 0
        out0 = output_lens[0] if n > 0 else 0
        eff_sys = max(0, min(sys_tokens, budget - u0 - out0))
        if u0 + out0 > budget:
            out0 = max(1, budget - u0)

        prev_total = 0
        for k in range(n):
            u_k = input_lens[k]
            if k == 0:
                total_input = eff_sys + u_k
                out_k = out0
            else:
                total_input = prev_total + eff_outputs[k - 1] + u_k
                # Input-overflow: no room for even one output token -> stop here.
                if total_input >= budget:
                    break
                out_k = output_lens[k]
                # Output-overflow: clamp to remaining room, keep the turn.
                if total_input + out_k > budget:
                    out_k = budget - total_input

            eff_inputs.append(u_k)
            eff_outputs.append(out_k)
            totals.append(total_input)
            prev_total = total_input

        return eff_sys, eff_inputs, eff_outputs, totals

    def _init_generation(self, config: DataConfig) -> None:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for conversation replay.")

        cr_config = config.conversation_replay
        if cr_config is None:
            raise ValueError("conversation_replay config is required.")
        self.cr_config: ConversationReplayConfig = cr_config

        # Tokenizer vocab size for random token generation
        hf_tokenizer = self.tokenizer.get_tokenizer()
        if hasattr(hf_tokenizer, "vocab_size") and hf_tokenizer.vocab_size is not None:
            self.vocab_size: int = hf_tokenizer.vocab_size
        elif hasattr(hf_tokenizer, "get_vocab") and callable(hf_tokenizer.get_vocab):
            self.vocab_size = len(hf_tokenizer.get_vocab())
        else:
            try:
                self.vocab_size = len(hf_tokenizer)
            except TypeError as e:
                raise ValueError("Cannot determine tokenizer vocabulary size.") from e
        if self.vocab_size <= 0:
            raise ValueError(f"Tokenizer vocabulary size must be positive, got {self.vocab_size}.")

        # Seeded RNG for deterministic generation
        self.rng = np.random.default_rng(self.cr_config.seed)
        self.max_model_len = self.cr_config.max_model_len or 225000

        # Cache for the currently active stage's shared system prompt
        self._current_stage_id: Optional[int] = None
        self._current_shared_prompt: Optional[str] = None

        self.blueprints: List[ConversationBlueprint] = []

    def _get_or_generate_shared_prompt(self, stage_id: int) -> str:
        """Get or generate the shared system prompt prefix for a specific stage using a derived stable seed."""
        if self._current_shared_prompt is None or self._current_stage_id != stage_id:
            if stage_id == 0:
                # Stage 0 draws from the shared global self.rng at startup. This is
                # deterministic within a run (seed -> identical blueprints in one
                # build, and identical across the closed-loop and open-loop
                # generators that share this builder). Cross-version seed stability is
                # not a goal, so the exact draw sequence may change between versions.
                self._current_shared_prompt = self._generate_random_token_text(self.cr_config.shared_system_prompt_len)
            else:
                # Later stages derive their seed stably from the base seed and stage ID.
                # This happens dynamically at runtime (post-setup) and uses local isolated RNGs.
                seed_str = f"{self.cr_config.seed}_stage_{stage_id}"
                hash_digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
                derived_seed = int.from_bytes(hash_digest[:4], byteorder="little")
                local_rng = np.random.default_rng(derived_seed)

                self._current_shared_prompt = self._generate_random_token_text(
                    self.cr_config.shared_system_prompt_len, rng=local_rng
                )
            self._current_stage_id = stage_id
        return self._current_shared_prompt

    def _build_system_prompt(self, shared_prompt: str, dynamic_prompt: str) -> str:
        """Combine shared prompt prefix and dynamic prompt suffix with a space."""
        return f"{shared_prompt} {dynamic_prompt}" if dynamic_prompt else shared_prompt

    def _sample_distribution(self, dist: Distribution, count: int) -> List[int]:
        """Sample ``count`` values from a Distribution."""
        arr = sample_from_distribution(dist, count, rng=self.rng)
        return [int(v) for v in arr]

    def _generate_random_token_text(self, num_tokens: int, rng: Optional[np.random.Generator] = None) -> str:
        """Generate random text that is approximately ``num_tokens`` long."""
        if num_tokens <= 0:
            return ""
        assert self.tokenizer is not None
        hf_tokenizer = self.tokenizer.get_tokenizer()
        use_rng = rng if rng is not None else self.rng
        token_ids = use_rng.integers(0, self.vocab_size, size=num_tokens).tolist()
        return str(hf_tokenizer.decode(token_ids, skip_special_tokens=True))

    def _build_conversations(self) -> None:
        """Pre-generate all conversation blueprints deterministically."""
        cfg = self.cr_config
        assert cfg.num_conversations is not None, "num_conversations must be resolved before build"
        n = cfg.num_conversations

        # Sample per-conversation parameters
        if cfg.turns_per_conversation is not None:
            turn_counts = self._sample_distribution(cfg.turns_per_conversation, n)
        else:
            turn_counts = [10] * n  # default fallback

        if cfg.dynamic_system_prompt_len is not None:
            dynamic_lens = self._sample_distribution(cfg.dynamic_system_prompt_len, n)
        else:
            dynamic_lens = [0] * n

        # Generate shared system prompt once (for stage 0)
        shared_prompt_text = self._get_or_generate_shared_prompt(0)

        total_turns = sum(turn_counts)
        logger.info(
            "Building %d conversations (%d total turns, shared prompt %d tokens)",
            n,
            total_turns,
            cfg.shared_system_prompt_len,
        )

        # Sample all turn-level parameters at once for efficiency
        if cfg.input_tokens_per_turn is not None:
            all_input_lens = self._sample_distribution(cfg.input_tokens_per_turn, total_turns)
        else:
            all_input_lens = [512] * total_turns

        if cfg.output_tokens_per_turn is not None:
            all_output_lens = self._sample_distribution(cfg.output_tokens_per_turn, total_turns)
        else:
            all_output_lens = [256] * total_turns

        if cfg.tool_call_latency_sec is not None:
            # Sample latencies as floats (seconds); re-use the same distribution
            # machinery but convert from the integer output to float seconds.
            all_tool_latencies: List[float] = [
                float(v) for v in self._sample_distribution(cfg.tool_call_latency_sec, total_turns)
            ]
        else:
            all_tool_latencies = []

        # Build each conversation. Lengths are resolved to fit the context window by
        # arithmetic FIRST (ignore_eos pins outputs, so the whole growing prefix is
        # known), then text is generated only at those effective lengths -- content
        # that would not fit is never sampled, decoded, or retained.
        turn_offset = 0
        shared_len = cfg.shared_system_prompt_len
        for conv_id in range(n):
            num_turns = turn_counts[conv_id]

            sampled_input_lens = all_input_lens[turn_offset : turn_offset + num_turns]
            sampled_output_lens = all_output_lens[turn_offset : turn_offset + num_turns]
            turn_tool_latencies = all_tool_latencies[turn_offset : turn_offset + num_turns] if all_tool_latencies else []
            turn_offset += num_turns

            sampled_sys = shared_len + dynamic_lens[conv_id]
            eff_sys, eff_input_lens, eff_output_lens, turn_totals = self._plan_within_budget(
                sampled_sys, list(sampled_input_lens), list(sampled_output_lens)
            )
            eff_num_turns = len(eff_input_lens)
            turn_tool_latencies = turn_tool_latencies[:eff_num_turns]

            # Split the effective system budget across the fixed shared prefix and the
            # per-conversation dynamic suffix. The dynamic part absorbs the clamp first
            # (keeping the shared prefix byte-identical across conversations for prefix
            # cache hits); only when eff_sys < shared_len do we shorten the shared text
            # for this conversation.
            eff_shared_len = min(shared_len, eff_sys)
            eff_dynamic_len = max(0, eff_sys - shared_len)
            shared_text_for_conv = (
                shared_prompt_text
                if eff_shared_len == shared_len
                else self._generate_random_token_text(eff_shared_len)
            )
            dynamic_text = self._generate_random_token_text(eff_dynamic_len)
            system_prompt = self._build_system_prompt(shared_text_for_conv, dynamic_text)

            # Generate turn prompts at their effective (unclamped) input lengths.
            turn_prompts: List[str] = [self._generate_random_token_text(tlen) for tlen in eff_input_lens]

            bp = ConversationBlueprint(
                conversation_id=conv_id,
                num_turns=eff_num_turns,
                system_prompt=system_prompt,
                system_prompt_tokens=eff_sys,
                dynamic_system_prompt=dynamic_text,
                turn_prompts=turn_prompts,
                turn_input_lens=eff_input_lens,
                turn_output_lens=eff_output_lens,
                turn_tool_call_latencies=turn_tool_latencies,
                turn_total_input_tokens=turn_totals,
            )
            self.blueprints.append(bp)


class ConversationReplayDataGenerator(DataGenerator, LazyLoadDataMixin, _ConversationBlueprintBuilder):
    """Generates synthetic multi-turn conversations from distribution configs.

    Each conversation has:
    - A two-part system prompt (shared prefix + dynamic per-conversation suffix)
    - N turns with independently sampled input/output token lengths
    - Sequential turn enforcement via LocalUserSession

    Conversations are dispatched round-robin across workers using
    preferred_worker_id for affinity. When all turns of a conversation are
    exhausted, the session resets and replays from the beginning (recycling).
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
    ) -> None:
        super().__init__(api_config, config, tokenizer)
        self._init_generation(config)

        # Build conversation blueprints, then one LocalUserSession per slot. Session
        # creation is a separate loop (not interleaved into _build_conversations) so the
        # shared builder stays dispatch-agnostic; LocalUserSession.__init__ consumes no
        # RNG, so blueprint generation is byte-identical to the interleaved form.
        self.user_sessions: List[LocalUserSession] = []
        self._build_conversations()
        for bp in self.blueprints:
            self.user_sessions.append(
                self._new_session(
                    user_session_id=f"conv_{bp.conversation_id}",
                    context=bp.system_prompt,
                    system_prompt=bp.system_prompt,
                )
            )

        logger.info(
            "ConversationReplayDataGenerator: %d conversations, %d total turns",
            len(self.blueprints),
            sum(bp.num_turns for bp in self.blueprints),
        )

    # -- BaseGenerator interface ------------------------------------------

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False

    def is_preferred_worker_requested(self) -> bool:
        return True

    # -- LazyLoadDataMixin interface --------------------------------------

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        conv_idx = data.data_index % len(self.blueprints)
        bp = self.blueprints[conv_idx]
        round_num = data.data_index // len(self.blueprints)
        turn_idx = round_num % bp.num_turns
        convo_num = round_num // bp.num_turns  # which conversation this slot is on

        # Re-prime if the session was cleared (e.g. LoadGenerator calls
        # LocalUserSession.clear_instances() between load stages). Without this,
        # subsequent stages would dispatch with empty session_context, losing the
        # system_prompt the conversation was built around.
        expected_session_id = self.user_sessions[conv_idx].user_session_id
        if expected_session_id not in LocalUserSession._instances:
            # Get or generate the shared system prompt for this stage.
            shared_prompt = self._get_or_generate_shared_prompt(data.stage_id)
            bp.system_prompt = self._build_system_prompt(shared_prompt, bp.dynamic_system_prompt)

            self.user_sessions[conv_idx] = self._new_session(
                user_session_id=expected_session_id,
                context=bp.system_prompt,
            )
            logger.debug("Slot %d: refreshed system prompt and re-primed session %s", conv_idx, expected_session_id)

        # Closed-loop replenishment: when a conversation finishes all its turns,
        # reset the session so the slot immediately starts a fresh conversation.
        if turn_idx == 0 and round_num > 0:
            self.user_sessions[conv_idx] = self._new_session(
                user_session_id=f"slot_{conv_idx}_convo_{convo_num}",
                context=bp.system_prompt,
                system_prompt=bp.system_prompt,
            )
            logger.debug("Slot %d starting conversation %d", conv_idx, convo_num)

        latency = bp.turn_tool_call_latencies[turn_idx] if bp.turn_tool_call_latencies else 0.0
        return _ConversationReplayAPIData(
            prompt=bp.turn_prompts[turn_idx],
            max_tokens=bp.turn_output_lens[turn_idx],
            user_session_id=self.user_sessions[conv_idx].user_session_id,
            target_round=round_num,
            tool_call_latency_sec=latency,
        )

    # -- DataGenerator interface ------------------------------------------

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if not self.blueprints:
            return

        i = 0
        while True:
            conv_idx = i % len(self.blueprints)
            yield LazyLoadInferenceAPIData(
                data_index=i,
                preferred_worker_id=conv_idx,
            )
            i += 1

    # -- Internal ---------------------------------------------------------

    def _new_session(self, user_session_id: str, context: str, system_prompt: str = "") -> LocalUserSession:
        session = LocalUserSession(
            user_session_id=user_session_id,
            context=context,
            system_prompt=system_prompt,
            tokenizer=self.tokenizer,
            max_model_len=self.max_model_len,
        )
        LocalUserSession._instances[user_session_id] = session
        return session


class ConversationSessionGenerator(ReplayGraphSessionGeneratorBase, _ConversationBlueprintBuilder):
    """Open-loop synthetic multi-turn conversations as graph-backed sessions.

    Shares blueprint generation with ``ConversationReplayDataGenerator`` (via
    ``_ConversationBlueprintBuilder``) so content is identical for a given config +
    seed, but dispatches through the session runtime instead of per-turn requests.

    Each conversation maps onto a linear ``ReplayGraph`` (turn k = one
    ``GraphEvent`` whose sole predecessor is turn k-1). Subclasses the graph session
    runtime, which supplies the whole ``SessionGenerator`` contract (session pool
    dispatch, worker affinity, lazy materialization, cross-process completion
    tracking, ``SessionLifecycleMetric`` recording). The only hook we implement is
    ``_build_session``.

    Driven by ``load.type: trace_session_replay`` through
    ``LoadGenerator.run_session_stage``: whole conversations arrive at
    ``session_rate`` conversations/sec with ``concurrent_sessions: 0`` (unbounded =
    true open loop). The load generator records a ``SessionLifecycleMetric`` per
    conversation, so achieved sessions/s and session duration are measured natively.
    Turn serialization is enforced by the graph runtime (turn k awaits turn k-1's
    recorded output); simulated tool-call latency maps to ``GraphEvent.wait_ms``
    (the runtime waits that long after the predecessor completes and before
    dispatching the turn, so it lands in the session duration).
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional[SyncManager] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        # replay_config=None: this is not a trace replay. All trace KV-replay
        # features (output substitution, random-session-id injection, tool-call
        # mitigation, wait-time capping) stay off, so requests are the synthetic
        # messages we build, sent verbatim.
        super().__init__(
            api_config,
            config,
            tokenizer,
            mp_manager=mp_manager,
            base_seed=base_seed,
            num_workers=num_workers,
            replay_config=None,
        )
        self._init_generation(config)
        self._build_conversations()

        if not self.blueprints:
            raise ValueError("ConversationSessionGenerator produced no conversation blueprints.")

        # One session slot per planned arrival across the whole run. run_session_stage
        # advances a run-wide cursor and draws num_sessions distinct indices per stage,
        # so the corpus must hold sum(num_sessions) slots. num_conversations is the total
        # arrival count (auto-sized by the config validator when unset); slots recycle the
        # blueprint pool by index % len(blueprints).
        total_arrivals = self.cr_config.num_conversations
        assert total_arrivals is not None, "num_conversations must be resolved before init"
        session_ids = [f"conv_{i}" for i in range(total_arrivals)]
        self.initialize_sessions_lazy(session_ids)

        logger.info(
            "ConversationSessionGenerator: %d blueprints, %d session slots (arrivals)",
            len(self.blueprints),
            total_arrivals,
        )

    # -- SessionGenerator hook --------------------------------------------

    def _build_session(self, session_index: int) -> Optional[ReplaySession]:
        """Build one conversation's linear ReplayGraph on demand.

        Recycles the blueprint pool by ``index % len(blueprints)`` for content
        variety across arrivals.
        """
        bp = self.blueprints[session_index % len(self.blueprints)]
        session_id = self._session_ids[session_index]
        graph = self._blueprint_to_graph(bp, session_id)
        return ReplaySession(
            session_id=session_id,
            source_id="conversation_replay",
            session_index=session_index,
            graph=graph,
        )

    def _blueprint_to_graph(self, bp: ConversationBlueprint, session_id: str) -> ReplayGraph:
        """Map a conversation blueprint onto a linear ReplayGraph with a growing,
        byte-stable conversation prefix (so vLLM's prefix cache is reused turn over
        turn, which is what session-affinity / prefix-aware routing wins on).

        Turn k is one GraphEvent whose sole predecessor is turn k-1. The runtime
        serialises turns (turn k awaits turn k-1's recorded output). Tool-call
        latency maps to wait_ms.

        Each turn resends the whole conversation so far, reconstructed from the
        registry rather than rebuilt by this builder, so the leading bytes are
        provably identical to what the predecessor put on the wire (== what vLLM
        cached). We reference ONLY turn k-1 (O(1) per turn), because turn k-1's
        recorded INPUT already chains the entire history (system + u0 + a0 + ... +
        u_{k-1}); its recorded OUTPUT is a_{k-1}. So turn k's assembled input is:

            [ shared -> turn_{k-1}.input ]   # system + u0 + a0 + ... + u_{k-1}
            [ output -> turn_{k-1}.output ]  # a_{k-1}, the tokens vLLM actually returned
            [ unique  = u_k ]                # the only genuinely new content this turn

        i.e. turn k grows the prefix by exactly (a_{k-1} + u_k). The messages list
        below is a builder-side FALLBACK copy (assistant slots are empty
        placeholders); the wire truth comes from substitution in
        _build_messages_with_substitution when the registry records exist. If a
        record is missing (e.g. predecessor served on another worker), substitution
        falls back to this recorded copy, so the graph stays well-formed.

        Turn 0 is the root: no predecessor, no segments, sent verbatim (system? + u0).

        The blueprint is already bounded to the context window at build time (see
        ``_plan_within_budget``): ``num_turns`` is the effective (post-stop) turn count,
        ``turn_output_lens`` are effective (post-clamp) outputs, and
        ``turn_total_input_tokens[k]`` is the exact cumulative input for turn k. So this
        method is pure assembly -- no budget checks, no tokenizer calls.
        """
        events: Dict[str, GraphEvent] = {}
        prev_id: Optional[str] = None
        prev_input_msg_count = 0
        prev_input_messages: List[Dict[str, Any]] = []

        def _in_len(idx: int) -> int:
            return bp.turn_input_lens[idx] if idx < len(bp.turn_input_lens) else 0

        for k in range(bp.num_turns):
            event_id = f"turn_{k}"
            user_msg = {"role": "user", "content": bp.turn_prompts[k]}
            u_k_tokens = _in_len(k)
            out_len = bp.turn_output_lens[k]
            total_input_tokens = bp.turn_total_input_tokens[k]

            messages: List[Dict[str, Any]]
            segments: List[InputSegment]
            if k == 0:
                messages = []
                if bp.system_prompt:
                    messages.append({"role": "system", "content": bp.system_prompt})
                messages.append(user_msg)
                segments = []
            else:
                # Fallback copy: predecessor's full input + an assistant placeholder
                # for a_{k-1} + the new user turn. Substitution overwrites the first
                # two groups from the registry; the placeholder never reaches the
                # wire un-substituted because turn k awaits turn k-1's completion.
                a_prev_tokens = bp.turn_output_lens[k - 1]
                assistant_ph = {"role": "assistant", "content": ""}
                messages = list(prev_input_messages) + [assistant_ph, user_msg]
                segments = [
                    # system + u0 + a0 + ... + u_{k-1}, pulled from turn_{k-1}'s
                    # recorded input (already chains all prior history -> O(1)).
                    InputSegment(
                        type="shared",
                        message_count=prev_input_msg_count,
                        token_count=bp.turn_total_input_tokens[k - 1],
                        source_event_id=prev_id,
                    ),
                    # a_{k-1}: the real tokens vLLM returned, from turn_{k-1}'s output.
                    # ignore_eos pins its length to the effective turn_output_lens[k-1].
                    InputSegment(
                        type="output",
                        message_count=1,
                        token_count=a_prev_tokens,
                        source_event_id=prev_id,
                    ),
                    # u_k: the only new content this turn.
                    InputSegment(type="unique", message_count=1, token_count=u_k_tokens),
                ]

            latency_sec = bp.turn_tool_call_latencies[k] if bp.turn_tool_call_latencies else 0.0
            wait_ms = int(latency_sec * 1000)

            call = GraphCall(
                call_id=f"{session_id}:{event_id}",
                model="",  # resolved from api_config by the client
                messages=messages,
                expected_output="",
                input_segments=segments,
                total_input_tokens=total_input_tokens,
                expected_output_tokens=out_len,
                temperature=None,
                max_tokens_recorded=out_len,
                tool_definitions=None,
                expected_output_is_tool_call=False,
            )
            events[event_id] = GraphEvent(
                event_id=event_id,
                call=call,
                predecessor_event_ids=[prev_id] if prev_id is not None else [],
                predecessor_dependency_types={prev_id: "output"} if prev_id is not None else {},
                wait_ms=wait_ms,
                t_start_ms=0,
                t_end_ms=0,
            )

            prev_input_messages = messages
            prev_input_msg_count = len(messages)
            prev_id = event_id

        return ReplayGraph(
            events=events,
            root_event_ids=["turn_0"],
            source_file="conversation_replay",
        )
