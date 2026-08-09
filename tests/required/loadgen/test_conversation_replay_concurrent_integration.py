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
"""End-to-end integration test for closed-loop (concurrent) conversation replay.

Drives the real LoadGenerator.run() through a forked Worker with a
ConversationReplayDataGenerator on a CONCURRENT load type — the closed-loop model, where
each slot recycles one conversation after another (infinite round-robin, sized by the
stage's num_requests). This is the counterpart to test_conversation_replay_integration.py
(open-loop). It also proves the session round-gate does NOT deadlock closed-loop: the
recycled sessions pass an ever-growing target_round while each fresh session starts its
_current_round at 0 (the B2 session-relative gate handles this).

The client writes each request to an mp.Queue (it runs in the child process); the main
process drains and asserts after the run.
"""

import multiprocessing as mp
import re
from collections import defaultdict
from queue import Empty
from typing import List, Optional, Tuple

import pytest
from unittest.mock import MagicMock

from inference_perf.apis import InferenceAPIData
from inference_perf.apis.user_session import UserSessionCompletionAPIData
from inference_perf.client.modelserver.base import ModelServerClient
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.config import (
    APIConfig,
    APIType,
    ConcurrentLoadStage,
    ConversationReplayConfig,
    DataConfig,
    DataGenType,
    Distribution,
    LoadConfig,
    LoadType,
)
from inference_perf.datagen.conversation_replay_datagen import ConversationReplayDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    hf = MagicMock()
    hf.vocab_size = 1000
    hf.decode = MagicMock(side_effect=lambda ids, **kw: f"tok_{len(ids)}")
    hf.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"tok_{len(ids)}" for ids in batch])
    hf.encode = MagicMock(side_effect=lambda text, **kw: [0, 0, 0])
    tok.get_tokenizer.return_value = hf
    tok.count_tokens = MagicMock(side_effect=lambda text, **kw: len(text.split()) if isinstance(text, str) else 0)
    return tok


def _make_generator(num_conversations: int, fixed_turns: int) -> ConversationReplayDataGenerator:
    api_config = APIConfig(type=APIType.Completion)
    cr_config = ConversationReplayConfig(
        seed=42,
        num_conversations=num_conversations,
        shared_system_prompt_len=5,
        turns_per_conversation=Distribution(type="fixed", min=fixed_turns, max=fixed_turns, mean=fixed_turns, std_dev=0),
        input_tokens_per_turn=Distribution(type="fixed", min=3, max=3, mean=3, std_dev=0),
        output_tokens_per_turn=Distribution(type="fixed", min=3, max=3, mean=3, std_dev=0),
    )
    data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)
    return ConversationReplayDataGenerator(api_config, data_config, _mock_tokenizer())


class _TurnRecordingClient(ModelServerClient):
    """Fake client that drives the turn lifecycle and records each request to a queue.

    Mirrors the open-loop integration test's client: runs to_request_body (gates on the
    session round, builds the prompt with accumulated history), records
    (session_id, round, prompt), then advances the round with a uniquely-marked response.
    """

    def __init__(self, queue: "mp.Queue[Tuple[str, int, str]]") -> None:
        self.api_config = APIConfig(type=APIType.Completion)
        self.timeout = None
        self._queue = queue

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        payload = await data.to_request_body("model", 64, False, False)
        prompt = payload["prompt"]
        assert isinstance(data, UserSessionCompletionAPIData)
        session_id = data.user_session_id
        self._queue.put((session_id, data.target_round, prompt))
        data.user_session.update_context(prompt + f" RESP_{session_id}_R{data.target_round}")

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        raise NotImplementedError


class TestClosedLoopConversationReplayIntegration:
    @pytest.mark.asyncio
    async def test_drives_recycling_slots_end_to_end(self) -> None:
        mp.set_start_method("fork", force=True)

        # 4 slots, 2 turns each; 16 requests = 2 full conversations per slot (recycled).
        num_requests, concurrency_level, fixed_turns = 16, 4, 2
        datagen = _make_generator(num_conversations=concurrency_level, fixed_turns=fixed_turns)

        # Replicate main.py's concurrent->rate/duration rewrite so run_stage sizes the run
        # from num_requests (rate*duration == num_requests).
        stage = ConcurrentLoadStage(num_requests=num_requests, concurrency_level=concurrency_level)
        stage.duration = 1
        stage.rate = num_requests
        load_config = LoadConfig(
            type=LoadType.CONCURRENT,
            stages=[stage],
            num_workers=1,
            interval=0,
            worker_max_concurrency=concurrency_level,
        )
        loadgen = LoadGenerator(datagen, load_config)
        queue: "mp.Queue[Tuple[str, int, str]]" = mp.Queue()
        client = _TurnRecordingClient(queue)

        await loadgen.run(client)
        await loadgen.stop()

        turns_by_session: dict[str, list[tuple[int, str]]] = defaultdict(list)
        while True:
            try:
                session_id, rnd, prompt = queue.get_nowait()
                turns_by_session[session_id].append((rnd, prompt))
            except Empty:
                break

        total_requests = sum(len(t) for t in turns_by_session.values())

        # 1. Closed-loop ran exactly num_requests turns (no deadline pruning; proves no deadlock).
        assert total_requests == num_requests, f"expected {num_requests} turns, got {total_requests}"

        # 2. Session ids are the closed-loop seed slots (conv_N) or recycled slots (slot_N_convo_M).
        assert all(re.fullmatch(r"conv_\d+|slot_\d+_convo_\d+", sid) for sid in turns_by_session), (
            f"unexpected session ids: {list(turns_by_session)}"
        )

        # 3. Within each recycled session, turns arrive in order and history accumulates
        #    (each later turn's prompt carries the prior turn's marked response).
        for session_id, turns in turns_by_session.items():
            local_rounds = [r for r, _ in turns]
            assert local_rounds == sorted(local_rounds), f"{session_id} rounds out of order: {local_rounds}"
            for i in range(1, len(turns)):
                _, prompt = turns[i]
                marker = f"RESP_{session_id}_R{turns[i - 1][0]}"
                assert marker in prompt, (
                    f"{session_id} turn {i} prompt missing prior turn's response ({marker!r}); "
                    f"history did not accumulate.\n  prompt: {prompt!r}"
                )
