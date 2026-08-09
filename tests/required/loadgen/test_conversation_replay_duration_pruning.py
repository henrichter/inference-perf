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
"""End-to-end test that open-loop conversation replay prunes its serialized tail.

Open-loop arrivals enqueue every turn of every conversation up front, but turns are
round-gated: turn K+1 waits for turn K's response. A conversation that arrives late
therefore keeps walking its remaining turns well past the stage `duration`. This test
drives a real forked Worker with a client that sleeps per turn (so conversations cannot
finish within the window) and asserts that at `start + duration` the stage:
  1. stops early — fewer turns are dispatched than the generator planned, and
  2. is still recorded COMPLETED (a graceful prune, not a FAILED timeout).
"""

import multiprocessing as mp
import time
from queue import Empty
from typing import List, Optional, Tuple

import pytest
from unittest.mock import MagicMock

from inference_perf.apis import InferenceAPIData
from inference_perf.apis.user_session import UserSessionCompletionAPIData
from inference_perf.client.modelserver.base import ModelServerClient
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.server_metrics.base import StageStatus
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


class _SlowTurnClient(ModelServerClient):
    """Fake client that drives the turn lifecycle but takes a fixed time per turn.

    The per-turn delay makes each conversation's serialized turns run long past the stage
    duration, so the pruning deadline must cut them off. Each completed turn is counted on
    a shared queue so the parent can assert how many actually ran.
    """

    def __init__(self, queue: "mp.Queue[Tuple[str, int]]", turn_delay: float) -> None:
        self.api_config = APIConfig(type=APIType.Completion)
        self.timeout = None
        self._queue = queue
        self._turn_delay = turn_delay

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        import asyncio

        await data.to_request_body("model", 64, False, False)
        assert isinstance(data, UserSessionCompletionAPIData)
        await asyncio.sleep(self._turn_delay)
        self._queue.put((data.user_session_id, data.target_round))
        data.user_session.update_context(f"resp_{data.user_session_id}_r{data.target_round}")

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        raise NotImplementedError


class TestConversationReplayDurationPruning:
    @pytest.mark.asyncio
    async def test_prunes_tail_and_completes_at_duration(self) -> None:
        mp.set_start_method("fork", force=True)

        # 2 arrivals/sec * 2s = 4 conversations, 8 turns each = 32 planned turns. With a
        # 0.3s per-turn delay, one conversation alone needs ~2.4s of serial work, so the
        # full plan cannot finish within the 2s duration — the tail must be pruned.
        rate, duration, fixed_turns, turn_delay = 2, 2, 8, 0.3
        datagen = _make_generator(num_conversations=4, fixed_turns=fixed_turns)
        load_config = LoadConfig(
            type=LoadType.POISSON,
            stages=[StandardLoadStage(rate=rate, duration=duration)],
            num_workers=2,
            interval=0,
        )
        loadgen = LoadGenerator(datagen, load_config)
        queue: "mp.Queue[Tuple[str, int]]" = mp.Queue()
        client = _SlowTurnClient(queue, turn_delay=turn_delay)

        start = time.perf_counter()
        await loadgen.run(client)
        elapsed = time.perf_counter() - start
        await loadgen.stop()

        completed_turns = 0
        while True:
            try:
                queue.get_nowait()
                completed_turns += 1
            except Empty:
                break

        datagen.init_stage(StandardLoadStage(rate=rate, duration=duration))
        planned = datagen.get_request_count()
        assert planned == 32, f"expected 32 planned turns, got {planned}"

        # 1. Pruned: fewer turns ran than were planned (the serialized tail was cut off).
        assert completed_turns < planned, f"expected fewer than {planned} turns to run (tail pruned), got {completed_turns}"
        assert completed_turns > 0, "no turns ran at all"

        # 2. Graceful: the stage ends near the duration, not after the full serial tail
        #    (which would be many multiples of `duration`). Generous upper bound absorbs
        #    the +1s queue head-start and drain settling.
        assert elapsed < duration + 5, f"stage did not stop near duration: {elapsed:.1f}s"

        # 3. The stage is recorded COMPLETED, not FAILED — hitting the duration is a
        #    graceful prune.
        assert loadgen.stage_runtime_info[0].status == StageStatus.COMPLETED
