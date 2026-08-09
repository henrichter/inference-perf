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
"""End-to-end integration test for open-loop conversation replay.

Drives the real LoadGenerator.run() through a forked Worker (num_workers=1, the
path real runs use) with a ConversationReplayDataGenerator and a fake
ModelServerClient that exercises the full to_request_body / update_context turn
lifecycle. This covers the integrated path — init_stage -> get_timer ->
ArrivalRepeatLoadTimer -> worker -> LocalUserSession round gating — that the datagen
unit tests do not. The client writes each request to an mp.Queue (it runs in the child
process); the main process drains and asserts after the run.
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
    ConversationReplayConfig,
    DataConfig,
    DataGenType,
    Distribution,
    LoadConfig,
    LoadType,
    StandardLoadStage,
)
from inference_perf.datagen.conversation_replay_datagen import ConversationReplayDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    hf = MagicMock()
    hf.vocab_size = 1000
    hf.decode = MagicMock(side_effect=lambda ids, **kw: f"tok_{len(ids)}")
    hf.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"tok_{len(ids)}" for ids in batch])
    # Short, fixed length so the max_model_len truncation branch in to_request_body never fires.
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

    process_request runs to_request_body (which gates on the session round and builds
    the prompt with accumulated history), writes (session_id, round, prompt) to the
    queue, then calls update_context with a uniquely-marked response — advancing the
    round so the session's next turn can proceed, as the real completion client does.
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
        # Advance the round and append a uniquely-marked response so the NEXT turn's
        # prompt provably contains this turn's output if history accumulated.
        data.user_session.update_context(prompt + f" RESP_{session_id}_R{data.target_round}")

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        raise NotImplementedError


class TestOpenLoopConversationReplayIntegration:
    @pytest.mark.asyncio
    async def test_drives_ordered_accumulating_turns_end_to_end(self) -> None:
        mp.set_start_method("fork", force=True)

        # 3 arrivals/sec * 2s = up to 6 conversations, 2 turns each = up to 12 requests.
        rate, duration, fixed_turns = 3, 2, 2
        datagen = _make_generator(num_conversations=4, fixed_turns=fixed_turns)
        load_config = LoadConfig(
            type=LoadType.POISSON,
            stages=[StandardLoadStage(rate=rate, duration=duration)],
            num_workers=1,
            interval=0,
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
        assert total_requests > 0, "no requests were dispatched"

        # 1. Open-loop fan-out: one distinct conv_open_* session per arrival.
        assert all(re.fullmatch(r"conv_open_\d+_\d+", sid) for sid in turns_by_session), (
            f"unexpected session ids: {list(turns_by_session)}"
        )

        # 2. Dispatched count is bounded by the generator's plan; the duration deadline may
        #    prune conversations that arrive near the end of the window, so it need not be exact.
        datagen.init_stage(StandardLoadStage(rate=rate, duration=duration))
        assert 0 < total_requests <= datagen.get_request_count()

        # 3. Per session: turns arrive in order (0,1,...) and each later turn's prompt
        #    contains the previous turn's marked response — i.e. round gating kept
        #    turns sequential and history accumulated across them.
        for session_id, turns in turns_by_session.items():
            rounds = [r for r, _ in turns]
            assert rounds == list(range(len(rounds))), f"{session_id} rounds not ordered 0..N-1: {rounds}"
            # Each turn after the first must carry the prior turn's marked response.
            for i in range(1, len(turns)):
                rnd, prompt = turns[i]
                marker = f"RESP_{session_id}_R{i - 1}"
                assert marker in prompt, (
                    f"{session_id} round {rnd} prompt missing prior turn's response "
                    f"({marker!r}); history did not accumulate.\n  prompt: {prompt!r}"
                )
