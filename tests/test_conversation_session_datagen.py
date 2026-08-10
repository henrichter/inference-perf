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
"""Tests for ConversationSessionGenerator (graph-backed open-loop session path)."""

from typing import Any
from dataclasses import replace

import pytest
from unittest.mock import MagicMock

from inference_perf.config import (
    APIConfig,
    APIType,
    ConversationReplayConfig,
    Distribution,
    DataConfig,
    DataGenType,
)
from inference_perf.datagen.conversation_replay_datagen import (
    ConversationReplayDataGenerator,
    ConversationSessionGenerator,
)
from inference_perf.datagen.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)
from inference_perf.datagen.otel_trace_to_replay_graph import RawCall, decompose_input
from inference_perf.datagen.replay_graph_types import ReplayMessage
from inference_perf.apis.base import SessionLifecycleMetric


def _make_mock_tokenizer(vocab_size: int = 32000) -> MagicMock:
    mock_tokenizer = MagicMock()
    mock_tokenizer.count_tokens.side_effect = lambda text, **kw: len(text.split()) * 10 if text.strip() else 0
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
    # encode returns one id per whitespace token (used only by any residual callers;
    # the generator now clamps by token-length arithmetic at build time).
    hf_tok.encode.side_effect = lambda text, **kwargs: list(range(len(text.split())))
    mock_tokenizer.get_tokenizer.return_value = hf_tok
    return mock_tokenizer


def _fixed_config(
    num_conversations: int = 3,
    fixed_turns: int = 3,
    tool_call_latency: float | None = None,
    api_type: APIType = APIType.Chat,
) -> tuple[APIConfig, DataConfig]:
    kwargs: dict[str, Any] = dict(
        seed=42,
        num_conversations=num_conversations,
        shared_system_prompt_len=10,
        turns_per_conversation=Distribution(type="fixed", min=fixed_turns, max=fixed_turns, mean=fixed_turns, std_dev=0),
        input_tokens_per_turn=Distribution(type="fixed", min=5, max=5, mean=5, std_dev=0),
        output_tokens_per_turn=Distribution(type="fixed", min=7, max=7, mean=7, std_dev=0),
    )
    if tool_call_latency is not None:
        kwargs["tool_call_latency_sec"] = Distribution(
            type="fixed", min=tool_call_latency, max=tool_call_latency, mean=tool_call_latency, std_dev=0
        )
    return APIConfig(type=api_type), DataConfig(
        type=DataGenType.ConversationReplay, conversation_replay=ConversationReplayConfig(**kwargs)
    )


def _make_gen(num_conversations: int = 3, fixed_turns: int = 3, tool_call_latency: float | None = None) -> ConversationSessionGenerator:
    api, dc = _fixed_config(num_conversations, fixed_turns, tool_call_latency)
    return ConversationSessionGenerator(api, dc, _make_mock_tokenizer())


class TestConversationSessionGenerator:
    def test_supported_apis_is_chat_not_completion(self) -> None:
        gen = _make_gen()
        apis = gen.get_supported_apis()
        assert APIType.Chat in apis
        assert APIType.Completion not in apis

    def test_session_count_equals_num_conversations(self) -> None:
        gen = _make_gen(num_conversations=7)
        assert gen.get_session_count() == 7

    def test_build_session_makes_linear_graph(self) -> None:
        gen = _make_gen(num_conversations=2, fixed_turns=4)
        session = gen._build_session(0)
        assert session is not None
        assert session.session_id == "conv_0"
        assert session.source_id == "conversation_replay"
        graph = session.graph
        assert graph.root_event_ids == ["turn_0"]
        assert len(graph.events) == 4
        assert graph.events["turn_0"].predecessor_event_ids == []
        for k in range(1, 4):
            e = graph.events[f"turn_{k}"]
            assert e.predecessor_event_ids == [f"turn_{k - 1}"]
            assert e.predecessor_dependency_types == {f"turn_{k - 1}": "output"}

    def test_first_turn_has_system_prompt(self) -> None:
        gen = _make_gen(num_conversations=1, fixed_turns=3)
        session = gen._build_session(0)
        assert session is not None
        # Turn 0 (root) is sent verbatim: system + first user turn.
        assert [m["role"] for m in session.graph.events["turn_0"].call.messages] == ["system", "user"]
        # Turn 1 carries the growing transcript: the system + prior user turn, the
        # assistant placeholder for turn 0's output (filled from the registry at
        # dispatch), then the new user turn.
        assert [m["role"] for m in session.graph.events["turn_1"].call.messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

    def test_transcript_grows_by_two_messages_per_turn(self) -> None:
        # Each turn appends exactly (prev assistant output + new user turn) => the
        # builder-side message list grows by 2 per turn after the root.
        gen = _make_gen(num_conversations=1, fixed_turns=4)
        session = gen._build_session(0)
        assert session is not None
        lengths = [len(session.graph.events[f"turn_{k}"].call.messages) for k in range(4)]
        # turn 0: [system, u0] = 2; then +2 each turn.
        assert lengths == [2, 4, 6, 8]

    def test_output_tokens_from_blueprint(self) -> None:
        gen = _make_gen(num_conversations=1, fixed_turns=3)
        bp = gen.blueprints[0]
        session = gen._build_session(0)
        assert session is not None
        for k in range(bp.num_turns):
            call = session.graph.events[f"turn_{k}"].call
            assert call.expected_output_tokens == bp.turn_output_lens[k]
            assert call.max_tokens_recorded == bp.turn_output_lens[k]

    def test_tool_call_latency_maps_to_wait_ms(self) -> None:
        gen = _make_gen(num_conversations=1, fixed_turns=3, tool_call_latency=2)
        session = gen._build_session(0)
        assert session is not None
        for k in range(3):
            assert session.graph.events[f"turn_{k}"].wait_ms == 2000

    def test_growing_prefix_segments_reference_only_predecessor(self) -> None:
        # Turn 0 is the verbatim root (no segments). Every later turn references
        # ONLY turn k-1 (O(1) chaining): a `shared` segment covering the whole prior
        # transcript, an `output` segment for turn k-1's returned tokens, and a
        # `unique` segment for the new user turn.
        gen = _make_gen(num_conversations=1, fixed_turns=4)
        session = gen._build_session(0)
        assert session is not None

        turn0 = session.graph.events["turn_0"].call
        assert turn0.input_segments == []
        assert turn0.tool_definitions is None
        assert turn0.expected_output_is_tool_call is False

        for k in range(1, 4):
            call = session.graph.events[f"turn_{k}"].call
            segs = call.input_segments
            assert [s.type for s in segs] == ["shared", "output", "unique"]
            shared, output, unique = segs
            # Both history segments reference ONLY the immediate predecessor.
            assert shared.source_event_id == f"turn_{k - 1}"
            assert output.source_event_id == f"turn_{k - 1}"
            assert unique.source_event_id is None
            # The `shared` segment covers turn k-1's ENTIRE recorded input; `output`
            # is exactly one message (a_{k-1}); `unique` is exactly the new user turn.
            assert output.message_count == 1
            assert unique.message_count == 1
            # shared count == (turn k-1's message count) == total(this turn) - 2.
            assert shared.message_count == len(call.messages) - 2
            # Segment message counts partition the full message list.
            assert sum(s.message_count for s in segs) == len(call.messages)

    def test_token_counts_are_accurate_and_grow(self) -> None:
        # Fixed config: system=10 tok, each user turn=5 tok, each assistant turn=7
        # tok (ignore_eos pins output length). Segment token_counts must sum to the
        # turn's total_input_tokens, and the total must grow by (a_{k-1} + u_k) = 12.
        SYS, U, A = 10, 5, 7
        gen = _make_gen(num_conversations=1, fixed_turns=4)
        session = gen._build_session(0)
        assert session is not None

        # turn 0: system + u0
        t0 = session.graph.events["turn_0"].call
        assert t0.total_input_tokens == SYS + U

        prev_total = t0.total_input_tokens
        for k in range(1, 4):
            call = session.graph.events[f"turn_{k}"].call
            shared, output, unique = call.input_segments
            # shared carries the predecessor's whole input token count.
            assert shared.token_count == prev_total
            # output = a_{k-1} length; unique = u_k length.
            assert output.token_count == A
            assert unique.token_count == U
            # Segment token counts sum to the turn's total.
            assert shared.token_count + output.token_count + unique.token_count == call.total_input_tokens
            # Grew by exactly a_{k-1} + u_k.
            assert call.total_input_tokens == prev_total + A + U
            prev_total = call.total_input_tokens


        # With num_conversations < total slots, slot idx maps to blueprint idx % pool.
        gen = _make_gen(num_conversations=3, fixed_turns=2)
        assert gen.blueprints[5 % len(gen.blueprints)] is gen.blueprints[2]

    def test_get_session_events_length_matches_turns(self) -> None:
        gen = _make_gen(num_conversations=2, fixed_turns=4)
        assert len(gen.get_session_events(0)) == 4

    def test_session_info_reports_turns(self) -> None:
        gen = _make_gen(num_conversations=2, fixed_turns=4)
        info = gen.get_session_info(0)
        assert info["session_id"] == "conv_0"
        assert info["num_events"] == 4

    def test_build_session_metric_on_clean_session(self) -> None:
        gen = _make_gen(num_conversations=1, fixed_turns=3)
        gen.get_session_events(0)  # builds session graph state
        gen.activate_session("conv_0")
        state = gen.get_session_state("conv_0")
        assert state is not None
        for k in range(3):
            state.completed_events.add(f"conv_0:turn_{k}")
        metric = gen.build_session_metric("conv_0", stage_id=0, start_time=100.0, end_time=142.5)
        assert isinstance(metric, SessionLifecycleMetric)
        assert metric.session_id == "conv_0"
        assert metric.stage_id == 0
        assert metric.duration_sec == pytest.approx(42.5)
        assert metric.num_events == 3
        assert metric.num_events_completed == 3

    def test_shares_blueprints_with_closed_loop_generator(self) -> None:
        # Same config + seed => identical blueprints across both generators (shared builder).
        api_chat, dc_chat = _fixed_config(num_conversations=3, fixed_turns=3, api_type=APIType.Chat)
        api_comp, dc_comp = _fixed_config(num_conversations=3, fixed_turns=3, api_type=APIType.Completion)
        sg = ConversationSessionGenerator(api_chat, dc_chat, _make_mock_tokenizer())
        cl = ConversationReplayDataGenerator(api_comp, dc_comp, _make_mock_tokenizer())
        assert [bp.turn_prompts for bp in sg.blueprints] == [bp.turn_prompts for bp in cl.blueprints]
        assert [bp.turn_output_lens for bp in sg.blueprints] == [bp.turn_output_lens for bp in cl.blueprints]


def _budget_config(
    *,
    max_model_len: int,
    fixed_turns: int,
    input_tokens: int,
    output_tokens: int,
    shared_system_prompt_len: int = 10,
    dynamic_system_prompt_len: int | None = None,
    num_conversations: int = 1,
) -> tuple[APIConfig, DataConfig]:
    """Config with an explicit max_model_len and distinct fixed input/output lengths,
    so tests can drive the input/output overflow branches deterministically."""
    kwargs: dict[str, Any] = dict(
        seed=42,
        num_conversations=num_conversations,
        shared_system_prompt_len=shared_system_prompt_len,
        max_model_len=max_model_len,
        turns_per_conversation=Distribution(type="fixed", min=fixed_turns, max=fixed_turns, mean=fixed_turns, std_dev=0),
        input_tokens_per_turn=Distribution(type="fixed", min=input_tokens, max=input_tokens, mean=input_tokens, std_dev=0),
        output_tokens_per_turn=Distribution(
            type="fixed", min=output_tokens, max=output_tokens, mean=output_tokens, std_dev=0
        ),
    )
    if dynamic_system_prompt_len is not None:
        kwargs["dynamic_system_prompt_len"] = Distribution(
            type="fixed",
            min=dynamic_system_prompt_len,
            max=dynamic_system_prompt_len,
            mean=dynamic_system_prompt_len,
            std_dev=0,
        )
    return APIConfig(type=APIType.Chat), DataConfig(
        type=DataGenType.ConversationReplay, conversation_replay=ConversationReplayConfig(**kwargs)
    )


class TestContextBudgetBounding:
    """Open-loop conversations must never put input+output past max_model_len.

    Two cases: input alone overflowing ends the conversation early; output
    overflowing clamps that turn's output. See _blueprint_to_graph.
    """

    def _budget(self, gen: ConversationSessionGenerator) -> int:
        return gen.max_model_len - gen._CONTEXT_SAFETY_BUFFER

    def test_blueprint_is_bounded_at_build_time(self) -> None:
        # The bounding happens in _build_conversations, not at graph-build: the
        # blueprint's effective turn count, per-turn totals, and system length all
        # already fit the budget, and turn_total_input_tokens is self-consistent.
        api, dc = _budget_config(max_model_len=5000, fixed_turns=20, input_tokens=1000, output_tokens=1000)
        gen = ConversationSessionGenerator(api, dc, _make_mock_tokenizer())
        budget = self._budget(gen)
        bp = gen.blueprints[0]
        assert bp.num_turns < 20  # stopped early
        assert len(bp.turn_output_lens) == bp.num_turns
        assert len(bp.turn_total_input_tokens) == bp.num_turns
        for k in range(bp.num_turns):
            assert bp.turn_total_input_tokens[k] + bp.turn_output_lens[k] <= budget
            # Recompute the cumulative input independently and compare.
            if k == 0:
                expected = bp.system_prompt_tokens + bp.turn_input_lens[0]
            else:
                expected = bp.turn_total_input_tokens[k - 1] + bp.turn_output_lens[k - 1] + bp.turn_input_lens[k]
            assert bp.turn_total_input_tokens[k] == expected

    def test_no_oversized_text_is_ever_generated(self) -> None:
        # The memory/CPU win: oversized content is never decoded. Every decode call
        # (random-token-text generation) requests at most `budget` tokens, even for a
        # config whose sampled dynamic system prompt is ~10x the window.
        from unittest.mock import patch

        api, dc = _budget_config(
            max_model_len=5000,
            fixed_turns=20,
            input_tokens=1000,
            output_tokens=1000,
            dynamic_system_prompt_len=50000,
        )
        tok = _make_mock_tokenizer()
        sizes: list[int] = []
        real_decode = tok.get_tokenizer().decode.side_effect

        def spy_decode(ids: Any, **kw: Any) -> Any:
            sizes.append(len(ids))
            return real_decode(ids, **kw)

        with patch.object(tok.get_tokenizer(), "decode", side_effect=spy_decode):
            gen = ConversationSessionGenerator(api, dc, tok)
        budget = self._budget(gen)
        assert sizes, "expected some text to be generated"
        assert max(sizes) <= budget, f"generated {max(sizes)} tokens, exceeds budget {budget}"

    def test_stops_on_input_overflow(self) -> None:
        # input=1000, output=1000 => each turn grows the prefix by ~2000 tokens.
        # budget = 5000 - 200 = 4800, so the prefix exceeds the budget after a few
        # turns and the conversation ends well before the sampled 20 turns.
        api, dc = _budget_config(max_model_len=5000, fixed_turns=20, input_tokens=1000, output_tokens=1000)
        gen = ConversationSessionGenerator(api, dc, _make_mock_tokenizer())
        session = gen._build_session(0)
        assert session is not None
        budget = self._budget(gen)
        assert 0 < len(session.graph.events) < 20
        for ev in session.graph.events.values():
            call = ev.call
            # Nothing on the wire may exceed the backend context window.
            assert call.total_input_tokens + call.expected_output_tokens <= budget
            # Input-overflow turns are dropped, not truncated: any built turn had
            # room for at least one output token.
            assert call.total_input_tokens < budget

    def test_clamps_output_on_output_overflow(self) -> None:
        # Small output normally, but the prefix grows until input+output just spills
        # past the budget on some turn while input still fits: that turn's output is
        # clamped rather than the turn being dropped.
        api, dc = _budget_config(max_model_len=5000, fixed_turns=20, input_tokens=500, output_tokens=500)
        gen = ConversationSessionGenerator(api, dc, _make_mock_tokenizer())
        session = gen._build_session(0)
        assert session is not None
        budget = self._budget(gen)

        clamped = []
        for k in range(len(session.graph.events)):
            call = session.graph.events[f"turn_{k}"].call
            assert call.total_input_tokens + call.expected_output_tokens <= budget
            # expected_output_tokens and max_tokens_recorded stay in lockstep.
            assert call.expected_output_tokens == call.max_tokens_recorded
            if call.expected_output_tokens < 500:
                clamped.append(k)
                # A clamped turn fills the budget exactly (uses all remaining room).
                assert call.total_input_tokens + call.expected_output_tokens == budget
        assert clamped, "expected at least one output-clamped turn"

        # The clamp must feed forward: turn k+1's `output` segment (a_{k}) reflects
        # the CLAMPED predecessor output, keeping the token chain exact.
        for k in clamped:
            nxt = f"turn_{k + 1}"
            if nxt in session.graph.events:
                out_seg = next(s for s in session.graph.events[nxt].call.input_segments if s.type == "output")
                assert out_seg.token_count == session.graph.events[f"turn_{k}"].call.expected_output_tokens

    def test_byte_stable_prefix_holds_after_bounding(self) -> None:
        # Even when the conversation is cut short by input-overflow, the turns that
        # ARE built must still form a byte-exact growing prefix.
        api, dc = _budget_config(max_model_len=6000, fixed_turns=20, input_tokens=800, output_tokens=800)
        gen = ConversationSessionGenerator(api, dc, _make_mock_tokenizer())
        assert len(gen._build_session(0).graph.events) < 20  # actually bounded
        wire = _simulate_dispatch_chain(gen)
        assert len(wire) >= 2
        for k in range(1, len(wire)):
            prev, cur = wire[k - 1], wire[k]
            assert cur[: len(prev)] == prev
            assert len(cur) == len(prev) + 2

    def test_system_prompt_clamped_so_turn0_fits(self) -> None:
        # A dynamic system prompt far larger than the whole context window must be
        # clamped at build time so turn 0 fits; turn 0 is still [system, user] and
        # within budget, and the blueprint stores the CLAMPED effective length.
        api, dc = _budget_config(
            max_model_len=5000,
            fixed_turns=3,
            input_tokens=100,
            output_tokens=100,
            shared_system_prompt_len=10,
            dynamic_system_prompt_len=50000,  # ~10x the whole window, as sampled
        )
        gen = ConversationSessionGenerator(api, dc, _make_mock_tokenizer())
        bp = gen.blueprints[0]
        # Sampled 50010 tokens, but the effective stored length is clamped to fit.
        assert bp.system_prompt_tokens < 50010
        assert bp.system_prompt_tokens <= self._budget(gen)
        session = gen._build_session(0)
        assert session is not None
        turn0 = session.graph.events["turn_0"].call
        assert [m["role"] for m in turn0.messages] == ["system", "user"]
        assert turn0.total_input_tokens + turn0.expected_output_tokens <= self._budget(gen)

    def test_no_bounding_when_it_fits(self) -> None:
        # Large max_model_len => every sampled turn is built and no output is clamped.
        api, dc = _budget_config(max_model_len=1_000_000, fixed_turns=6, input_tokens=100, output_tokens=100)
        gen = ConversationSessionGenerator(api, dc, _make_mock_tokenizer())
        session = gen._build_session(0)
        assert session is not None
        assert len(session.graph.events) == 6
        for k in range(6):
            assert session.graph.events[f"turn_{k}"].call.expected_output_tokens == 100


def _simulate_dispatch_chain(gen: ConversationSessionGenerator, session_index: int = 0) -> list[list[dict]]:
    """Drive the REAL substitution path turn-by-turn against a shared registry.

    Mirrors what the runtime does at dispatch: build each turn's
    SessionChatCompletionAPIData from the graph event, qualify segment
    source_event_ids with the session id, run the real
    ``_build_messages_with_substitution``, then record that turn's output back to
    the registry (as ``on_completion`` would) so the next turn can chain off it.

    Returns the reconstructed wire message list (as plain dicts) for each turn.
    This is the ground truth the model server would receive.
    """
    session = gen._build_session(session_index)
    assert session is not None
    sid = session.session_id
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    num_turns = len(session.graph.events)

    reconstructed: list[list[dict]] = []
    for k in range(num_turns):
        event = session.graph.events[f"turn_{k}"]
        gc = event.call
        # Qualify segment source ids the way the base runtime does before dispatch.
        qualified_segments = [
            replace(seg, source_event_id=f"{sid}:{seg.source_event_id}") if seg.source_event_id else seg
            for seg in gc.input_segments
        ]
        api_data = SessionChatCompletionAPIData(
            messages=[],  # unused by _build_messages_with_substitution
            max_tokens=gc.expected_output_tokens,
            tool_definitions=gc.tool_definitions,
            event_id=f"{sid}:turn_{k}",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=num_turns,
            predecessor_event_ids=[f"{sid}:{p}" for p in event.predecessor_event_ids],
            input_segments=qualified_segments,
            original_messages=[dict(m) for m in gc.messages],  # builder fallback copy
        )
        msgs = api_data._build_messages_with_substitution()
        # Normalise to plain {role, content} dicts for comparison.
        norm = [{"role": m["role"], "content": m.get("content", "")} for m in msgs]
        reconstructed.append(norm)

        # Record this turn's output + its assembled input, as on_completion does,
        # so turn k+1's shared/output segments resolve against real data.
        output_text = f"ASSISTANT_OUTPUT_turn_{k}"
        registry.record(
            f"{sid}:turn_{k}",
            output_text,
            norm,
            output_message={"role": "assistant", "content": output_text},
        )
    return reconstructed


class TestGrowingPrefixReconstruction:
    """End-to-end: the reconstructed wire transcript must be a byte-exact,
    monotonically growing prefix (what vLLM's prefix cache reuses turn over turn).
    """

    def test_each_turn_is_byte_exact_prefix_extension_of_previous(self) -> None:
        gen = _make_gen(num_conversations=1, fixed_turns=5)
        wire = _simulate_dispatch_chain(gen)

        for k in range(1, len(wire)):
            prev, cur = wire[k - 1], wire[k]
            # The new turn must START with the entire previous turn's messages,
            # byte-identical (this is the cache-hit prefix).
            assert cur[: len(prev)] == prev, (
                f"turn {k} is not a prefix-extension of turn {k - 1}:\n"
                f"prev={prev}\ncur[:len(prev)]={cur[: len(prev)]}"
            )
            # And it must grow by exactly two messages: the predecessor's assistant
            # output (a_{k-1}) + the new user turn (u_k).
            assert len(cur) == len(prev) + 2
            assert cur[len(prev)]["role"] == "assistant"
            assert cur[len(prev) + 1]["role"] == "user"

    def test_assistant_slots_hold_real_recorded_output(self) -> None:
        # Each assistant message in the reconstructed transcript must be the tokens
        # the predecessor turn actually "returned" (pulled from the registry), NOT a
        # builder-side placeholder. This is what makes the cached prefix match.
        gen = _make_gen(num_conversations=1, fixed_turns=4)
        wire = _simulate_dispatch_chain(gen)

        final = wire[-1]  # turn 3: [system, u0, a0, u1, a1, u2, a2, u3]
        assistant_contents = [m["content"] for m in final if m["role"] == "assistant"]
        assert assistant_contents == [
            "ASSISTANT_OUTPUT_turn_0",
            "ASSISTANT_OUTPUT_turn_1",
            "ASSISTANT_OUTPUT_turn_2",
        ]
        # No empty placeholder ever survives into the wire transcript.
        assert all(m["content"] != "" for m in final if m["role"] == "assistant")

    def test_prefix_bytes_are_stable_across_all_turns(self) -> None:
        # Stronger form: the assistant/user content for a given position never
        # changes once it first appears (frozen-registry guarantee).
        gen = _make_gen(num_conversations=1, fixed_turns=5)
        wire = _simulate_dispatch_chain(gen)

        seen: dict[int, dict] = {}
        for turn_msgs in wire:
            for pos, msg in enumerate(turn_msgs):
                if pos in seen:
                    assert seen[pos] == msg, f"message at position {pos} changed between turns"
                else:
                    seen[pos] = msg


def _raw_calls_from_wire(wire: list[list[dict]]) -> list[RawCall]:
    """Turn the dispatch-reconstructed message lists into otel RawCall objects.

    Each turn's RawCall gets that turn's FULL cumulative messages (as a real trace
    would record) and an out_message = the assistant answer this turn produced
    (ASSISTANT_OUTPUT_turn_k, injected by _simulate_dispatch_chain). This is exactly
    the shape otel's decompose_input expects, so we can ask the canonical decomposer
    to re-derive the segment structure our composer emitted.
    """
    calls: list[RawCall] = []
    for k, msgs in enumerate(wire):
        rc = RawCall(
            call_id=f"turn_{k}",
            trace_id="t",
            t_start_ms=k,
            t_end_ms=k,
            model="m",
            messages=[ReplayMessage(role=m["role"], text=m.get("content", "")) for m in msgs],
            out_message=ReplayMessage(role="assistant", text=f"ASSISTANT_OUTPUT_turn_{k}"),
            prompt_tokens=None,
            completion_tokens=None,
            temperature=None,
            max_tokens_recorded=None,
        )
        calls.append(rc)
    return calls


class TestComposerMatchesOtelDecomposer:
    """Our synthetic composer builds the segment structure directly; otel's
    decompose_input infers it by diffing a materialised trace. For a LINEAR
    conversation the two must agree on segment types, message counts, and the
    predecessor each history segment points at -- proving the hand-rolled fast
    path stays consistent with the canonical decomposer (guards against drift).
    """

    def test_linear_composition_matches_decomposition(self) -> None:
        gen = _make_gen(num_conversations=1, fixed_turns=5)

        # 1) Our composer's segments, per turn.
        session = gen._build_session(0)
        assert session is not None
        composer_segs = {
            f"turn_{k}": session.graph.events[f"turn_{k}"].call.input_segments
            for k in range(5)
        }

        # 2) The reconstructed cumulative message lists (real substitution path),
        #    turned into RawCalls, then decomposed by otel's canonical decomposer.
        wire = _simulate_dispatch_chain(gen)
        raw_calls = _raw_calls_from_wire(wire)

        for k in range(1, 5):
            # Linear chain: every prior turn is an ancestor; decompose_input picks
            # the longest-common-prefix predecessor, which for a linear conversation
            # is always turn k-1.
            predecessors = raw_calls[:k]
            pred_event_ids = [f"turn_{j}" for j in range(k)]
            decomposed = decompose_input(raw_calls[k], predecessors, pred_event_ids)

            composed = composer_segs[f"turn_{k}"]

            # Same segment TYPES in the same order.
            assert [s.type for s in decomposed] == [s.type for s in composed], (
                f"turn {k}: types differ\ncomposed={[s.type for s in composed]}\n"
                f"decomposed={[s.type for s in decomposed]}"
            )
            # Same per-segment MESSAGE COUNTS.
            assert [s.message_count for s in decomposed] == [s.message_count for s in composed], (
                f"turn {k}: message_counts differ\n"
                f"composed={[s.message_count for s in composed]}\n"
                f"decomposed={[s.message_count for s in decomposed]}"
            )
            # The shared + output history segments must both point at turn k-1.
            comp_shared, comp_output, _ = composed
            dec_shared = next(s for s in decomposed if s.type == "shared")
            dec_output = next(s for s in decomposed if s.type == "output")
            assert comp_shared.source_event_id == f"turn_{k - 1}"
            assert dec_shared.source_event_id == f"turn_{k - 1}"
            assert comp_output.source_event_id == f"turn_{k - 1}"
            assert dec_output.source_event_id == f"turn_{k - 1}"
