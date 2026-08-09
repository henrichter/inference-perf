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
"""Tests for ConversationReplayDataGenerator."""

from typing import Any, Generator, List, cast

import pytest
from unittest.mock import MagicMock
import numpy as np

from inference_perf.config import (
    APIConfig,
    APIType,
    ConversationReplayConfig,
    Distribution,
    DataConfig,
    DataGenType,
    StandardLoadStage,
)
from inference_perf.datagen.conversation_replay_datagen import (
    ConversationReplayDataGenerator,
    _ArrivalLazyData,
    _ConversationReplayAPIData,
)
from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.user_session import LocalUserSession
from inference_perf.loadgen.load_timer import ArrivalRepeatLoadTimer, ConstantLoadTimer
from inference_perf.utils.numeric.distribution import generate_distribution


@pytest.fixture(autouse=True)
def _clear_user_session_registry() -> Generator[None, None, None]:
    """Isolate LocalUserSession._instances across tests."""
    LocalUserSession.clear_instances()
    yield
    LocalUserSession.clear_instances()


def _make_mock_tokenizer(vocab_size: int = 32000) -> MagicMock:
    """Create a mock tokenizer with the expected interface."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.count_tokens.side_effect = lambda text, **kw: len(text.split()) * 10 if text.strip() else 0
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
    hf_tok.batch_decode.side_effect = lambda list_ids, **kwargs: [f"decoded_{ids}" for ids in list_ids]
    mock_tokenizer.get_tokenizer.return_value = hf_tok
    return mock_tokenizer


def _make_config(
    num_conversations: int = 5,
    seed: int = 42,
    shared_system_prompt_len: int = 100,
    turns_min: int = 3,
    turns_max: int = 5,
    turns_mean: float = 4,
) -> tuple[APIConfig, DataConfig]:
    api_config = APIConfig(type=APIType.Completion)
    cr_config = ConversationReplayConfig(
        seed=seed,
        num_conversations=num_conversations,
        shared_system_prompt_len=shared_system_prompt_len,
        dynamic_system_prompt_len=Distribution(type="normal", min=50, max=200, mean=100, std_dev=30),
        turns_per_conversation=Distribution(type="normal", min=turns_min, max=turns_max, mean=turns_mean, std_dev=1),
        input_tokens_per_turn=Distribution(type="normal", min=10, max=100, mean=50, std_dev=20),
        output_tokens_per_turn=Distribution(type="normal", min=10, max=100, mean=50, std_dev=20),
    )
    data_config = DataConfig(
        type=DataGenType.ConversationReplay,
        conversation_replay=cr_config,
    )
    return api_config, data_config


class TestConversationReplayDataGenerator:
    def test_init_creates_correct_number_of_conversations(self) -> None:
        api_config, data_config = _make_config(num_conversations=10)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert len(gen.blueprints) == 10
        assert len(gen.user_sessions) == 10

    def test_deterministic_with_same_seed(self) -> None:
        api_config, data_config = _make_config(seed=123)
        gen1 = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        api_config2, data_config2 = _make_config(seed=123)
        gen2 = ConversationReplayDataGenerator(api_config2, data_config2, _make_mock_tokenizer())

        assert len(gen1.blueprints) == len(gen2.blueprints)
        for bp1, bp2 in zip(gen1.blueprints, gen2.blueprints, strict=True):
            assert bp1.num_turns == bp2.num_turns
            assert bp1.turn_output_lens == bp2.turn_output_lens

    def test_different_seeds_produce_different_results(self) -> None:
        api_config1, data_config1 = _make_config(seed=1)
        gen1 = ConversationReplayDataGenerator(api_config1, data_config1, _make_mock_tokenizer())
        api_config2, data_config2 = _make_config(seed=2)
        gen2 = ConversationReplayDataGenerator(api_config2, data_config2, _make_mock_tokenizer())

        # At least some turn counts should differ
        turns1 = [bp.num_turns for bp in gen1.blueprints]
        turns2 = [bp.num_turns for bp in gen2.blueprints]
        assert turns1 != turns2

    def test_get_data_yields_lazy_load_with_preferred_worker(self) -> None:
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        data_iter = gen.get_data()
        items = [next(data_iter) for _ in range(9)]

        # First 3 items should cycle through conversations 0, 1, 2
        assert all(isinstance(item, LazyLoadInferenceAPIData) for item in items)
        assert items[0].preferred_worker_id == 0
        assert items[1].preferred_worker_id == 1
        assert items[2].preferred_worker_id == 2
        # Second round
        assert items[3].preferred_worker_id == 0

    def test_load_lazy_data_returns_user_session_data(self) -> None:
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)

        assert isinstance(result, _ConversationReplayAPIData)
        assert result.user_session == gen.user_sessions[0]
        assert result.target_round == 0

    def test_turn_recycling(self) -> None:
        """When data_index exceeds total turns, it wraps around."""
        api_config, data_config = _make_config(num_conversations=2, turns_min=3, turns_max=3, turns_mean=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        # Conversation 0 has 3 turns. data_index=0 -> round 0, turn 0
        # data_index=2 -> conv 0, round 1, turn 1
        # data_index=6 -> conv 0, round 3, turn 0 (recycled)
        lazy = LazyLoadInferenceAPIData(data_index=6, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.target_round == 3  # 6 // 2 = 3

    def test_requires_tokenizer(self) -> None:
        api_config, data_config = _make_config()
        with pytest.raises(ValueError, match="Tokenizer is required"):
            ConversationReplayDataGenerator(api_config, data_config, None)

    def test_requires_conversation_replay_config(self) -> None:
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(type=DataGenType.ConversationReplay)
        with pytest.raises(ValueError, match="conversation_replay config is required"):
            ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_is_preferred_worker_requested(self) -> None:
        api_config, data_config = _make_config()
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert gen.is_preferred_worker_requested() is True

    def test_user_session_ids(self) -> None:
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        ids = [s.user_session_id for s in gen.user_sessions]
        assert ids == ["conv_0", "conv_1", "conv_2"]

    def test_load_lazy_data_returns_conversation_replay_api_data(self) -> None:
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)

    def test_tool_call_latency_not_set_gives_zero(self) -> None:
        """Without tool_call_latency_sec, all latencies are 0."""
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.tool_call_latency_sec == 0.0

    def test_tool_call_latency_fixed_distribution(self) -> None:
        """Fixed tool call latency is sampled and stored per turn."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=2,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=3, max=3, mean=3, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            output_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            tool_call_latency_sec=Distribution(type="fixed", min=5, max=5, mean=5, std_dev=0),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        # All turns should have latency == 5.0
        for bp in gen.blueprints:
            assert len(bp.turn_tool_call_latencies) == bp.num_turns
            assert all(lat == 5.0 for lat in bp.turn_tool_call_latencies)

        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.tool_call_latency_sec == 5.0

    def test_load_lazy_data_regenerates_after_clear_instances(self) -> None:
        """After LoadGenerator clears the session registry between stages,
        load_lazy_data must regenerate the system_prompt for the slot."""
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        original_contexts = {bp.conversation_id: bp.system_prompt for bp in gen.blueprints}
        assert all(f"conv_{i}" in LocalUserSession._instances for i in range(3))

        # Simulate LoadGenerator's between-stage cleanup.
        LocalUserSession.clear_instances()
        assert LocalUserSession._instances == {}

        # First request after the clear should re-prime the slot it touches AND regenerate prompt.
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=1))

        assert "conv_1" in LocalUserSession._instances
        # It should be different from original
        assert LocalUserSession._instances["conv_1"].context != original_contexts[1]

    def test_system_prompt_regenerated_across_repeated_clears(self) -> None:
        """Across multiple stage transitions, each re-prime must regenerate a fresh system_prompt."""
        api_config, data_config = _make_config(num_conversations=3)
        # Use a tokenizer that returns different text each time to verify regeneration
        mock_tok = _make_mock_tokenizer()
        texts = [f"text_{i}" for i in range(100)]
        mock_tok.get_tokenizer().decode.side_effect = texts

        gen = ConversationReplayDataGenerator(api_config, data_config, mock_tok)

        last_contexts = {i: "" for i in range(3)}

        for stage_idx in range(3):
            LocalUserSession.clear_instances()
            for conv_idx in range(3):
                gen.load_lazy_data(
                    LazyLoadInferenceAPIData(data_index=conv_idx, preferred_worker_id=conv_idx, stage_id=stage_idx)
                )
                current_context = LocalUserSession._instances[f"conv_{conv_idx}"].context
                assert current_context != last_contexts[conv_idx]
                last_contexts[conv_idx] = current_context

    def test_load_lazy_data_does_not_replace_live_session(self) -> None:
        """When the registry still holds the session (mid-stage), load_lazy_data
        must not overwrite it — that would clobber accumulated turn context."""
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        live_session = LocalUserSession._instances["conv_0"]
        expected_context = f"{live_session.system_prompt} accumulated turn history"
        live_session.update_context(expected_context)

        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))

        assert LocalUserSession._instances["conv_0"] is live_session
        assert LocalUserSession._instances["conv_0"].context == expected_context

    def test_tool_call_latency_lognormal_distribution(self) -> None:
        """Lognormal tool call latencies vary across turns."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=3,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            output_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            tool_call_latency_sec=Distribution(type="lognormal", min=1, max=30, mean=8, std_dev=6),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        for bp in gen.blueprints:
            assert len(bp.turn_tool_call_latencies) == bp.num_turns
            # Should have variation (lognormal, not fixed)
            assert not all(lat == bp.turn_tool_call_latencies[0] for lat in bp.turn_tool_call_latencies)
            # All within bounds
            assert all(1 <= lat <= 30 for lat in bp.turn_tool_call_latencies)

    def test_reproducibility_across_runs_with_stages(self) -> None:
        """Verify that two independent runs with the same seed generate
        the same system prompt for the same stage."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=2,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=1, max=1, mean=1, std_dev=0),
            input_tokens_per_turn=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
            output_tokens_per_turn=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)

        def make_deterministic_mock_tok() -> MagicMock:
            mock_tok = MagicMock()
            hf_tok = MagicMock()
            hf_tok.vocab_size = 32000
            hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
            mock_tok.get_tokenizer.return_value = hf_tok
            return mock_tok

        # Run 1
        gen1 = ConversationReplayDataGenerator(api_config, data_config, make_deterministic_mock_tok())
        # Stage 0
        gen1.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        context1_stage0 = LocalUserSession._instances["conv_0"].context

        LocalUserSession.clear_instances()

        # Stage 1
        gen1.load_lazy_data(LazyLoadInferenceAPIData(data_index=2, preferred_worker_id=0, stage_id=1))
        context1_stage1 = LocalUserSession._instances["conv_0"].context

        # Isolate Run 2
        LocalUserSession.clear_instances()

        # Run 2
        gen2 = ConversationReplayDataGenerator(api_config, data_config, make_deterministic_mock_tok())
        # Stage 0
        gen2.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        context2_stage0 = LocalUserSession._instances["conv_0"].context

        LocalUserSession.clear_instances()

        # Stage 1
        gen2.load_lazy_data(LazyLoadInferenceAPIData(data_index=2, preferred_worker_id=0, stage_id=1))
        context2_stage1 = LocalUserSession._instances["conv_0"].context

        # Verify reproducibility across runs for Stage 0
        assert context1_stage0 == context2_stage0

        # Verify reproducibility across runs for Stage 1
        assert context1_stage1 == context2_stage1

        # Verify that prompts are DIFFERENT across stages within Run 2
        assert context2_stage0 != context2_stage1

    def test_shared_system_prompt_within_stage(self) -> None:
        """Verify that different conversations in the same stage have the same system prompt."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=2,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=1, max=1, mean=1, std_dev=0),
            input_tokens_per_turn=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
            output_tokens_per_turn=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)

        def make_deterministic_mock_tok() -> MagicMock:
            mock_tok = MagicMock()
            hf_tok = MagicMock()
            hf_tok.vocab_size = 32000
            hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
            mock_tok.get_tokenizer.return_value = hf_tok
            return mock_tok

        gen = ConversationReplayDataGenerator(api_config, data_config, make_deterministic_mock_tok())

        # Stage 0
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=0))

        context_conv0_s0 = LocalUserSession._instances["conv_0"].context
        context_conv1_s0 = LocalUserSession._instances["conv_1"].context
        assert context_conv0_s0 == context_conv1_s0

        # Transition to Stage 1
        LocalUserSession.clear_instances()
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=1))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=1))

        context_conv0_s1 = LocalUserSession._instances["conv_0"].context
        context_conv1_s1 = LocalUserSession._instances["conv_1"].context

        # Verify they are still identical in Stage 1
        assert context_conv0_s1 == context_conv1_s1
        # Verify Stage 1 prompt is different from Stage 0 prompt
        assert context_conv0_s1 != context_conv0_s0

    def test_unique_system_prompt_within_stage_after_clear(self) -> None:
        """Verify that different conversations with dynamic_system_prompt_len get unique system prompts after stage clear."""
        api_config, data_config = _make_config(num_conversations=2)

        def make_deterministic_mock_tok() -> MagicMock:
            mock_tok = MagicMock()
            hf_tok = MagicMock()
            hf_tok.vocab_size = 32000
            hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
            mock_tok.get_tokenizer.return_value = hf_tok
            return mock_tok

        gen = ConversationReplayDataGenerator(api_config, data_config, make_deterministic_mock_tok())

        # Stage 0 runtime priming
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=0))

        context_conv0_s0 = LocalUserSession._instances["conv_0"].context
        context_conv1_s0 = LocalUserSession._instances["conv_1"].context
        assert context_conv0_s0 != context_conv1_s0

        # Simulate stage transition by clearing instances and moving to Stage 1
        LocalUserSession.clear_instances()

        # Load lazy data for both conversations for Stage 1
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=1))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=1))

        context_conv0_s1 = LocalUserSession._instances["conv_0"].context
        context_conv1_s1 = LocalUserSession._instances["conv_1"].context

        # 1. The system prompts in Stage 1 should still be different from each other
        assert context_conv0_s1 != context_conv1_s1

        # 2. Stage 1 prompts should also be different from their Stage 0 versions (new stage prefix)
        assert context_conv0_s1 != context_conv0_s0
        assert context_conv1_s1 != context_conv1_s0

    def test_shared_system_prompt_cached_per_stage(self) -> None:
        """Verify that the shared system prompt is only generated once per stage and cached."""
        api_config, data_config = _make_config(num_conversations=2)
        mock_tok = _make_mock_tokenizer()

        # Track how many times decode is called.
        decode_count = 0

        def counting_decode(ids: Any, **kwargs: Any) -> str:
            nonlocal decode_count
            decode_count += 1
            return f"decoded_{ids}"

        mock_tok.get_tokenizer().decode.side_effect = counting_decode

        gen = ConversationReplayDataGenerator(api_config, data_config, mock_tok)

        # Reset decode count after initialization
        decode_count = 0

        # Simulate stage transition
        LocalUserSession.clear_instances()

        # Retrieve two sessions in the same stage
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=5))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=5))

        # Decode should have been called exactly once for the shared prompt across both slot re-primes.
        assert decode_count == 1

    def test_slot_reset_on_new_conversation(self) -> None:
        # On turn_idx==0 with round_num>0 the slot resets to a fresh session id.
        api_config, data_config = _make_config(num_conversations=2, turns_min=3, turns_max=3, turns_mean=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        # data_index=6 -> conv 0, round_num=3, turn_idx=0, convo_num=1 -> reset to slot_0_convo_1.
        result = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=6, preferred_worker_id=0))
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.user_session_id == "slot_0_convo_1"

    def test_init_stage_noop_does_not_crash_on_concurrent_stage(self) -> None:
        # mp_run calls init_stage for every stage; closed-loop must no-op on ConcurrentLoadStage
        # (not a StandardLoadStage) rather than assert-crash.
        from inference_perf.config import ConcurrentLoadStage

        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        gen.init_stage(ConcurrentLoadStage(num_requests=4, concurrency_level=2))  # must not raise
        assert gen.get_request_count() == 0


class TestDistributionExtensions:
    def test_lognormal_distribution(self) -> None:
        rng = np.random.default_rng(42)
        result = generate_distribution(
            min=10, max=1000, mean=100, std_dev=50, total_count=1000, dist_type="lognormal", rng=rng
        )
        assert len(result) == 1000
        assert all(10 <= v <= 1000 for v in result)

    def test_uniform_distribution(self) -> None:
        rng = np.random.default_rng(42)
        result = generate_distribution(min=10, max=100, mean=55, std_dev=0, total_count=1000, dist_type="uniform", rng=rng)
        assert len(result) == 1000
        assert all(10 <= v <= 100 for v in result)

    def test_fixed_distribution(self) -> None:
        result = generate_distribution(min=50, max=50, mean=50, std_dev=0, total_count=100, dist_type="fixed")
        assert len(result) == 100
        assert all(v == 50 for v in result)

    def test_normal_distribution_backward_compatible(self) -> None:
        """Default dist_type='normal' preserves existing behavior."""
        np.random.seed(42)
        result = generate_distribution(min=10, max=100, mean=50, std_dev=20, total_count=100)
        assert len(result) == 100
        assert all(10 <= v <= 100 for v in result)

    def test_seeded_rng_deterministic(self) -> None:
        rng1 = np.random.default_rng(99)
        result1 = generate_distribution(min=10, max=1000, mean=500, std_dev=100, total_count=50, dist_type="normal", rng=rng1)
        rng2 = np.random.default_rng(99)
        result2 = generate_distribution(min=10, max=1000, mean=500, std_dev=100, total_count=50, dist_type="normal", rng=rng2)
        assert list(result1) == list(result2)


class TestSlidingWindowTruncation:
    def test_sliding_window_truncation(self) -> None:
        from inference_perf.apis.user_session import LocalUserSession

        mock_tokenizer = MagicMock()
        # count_tokens returns 100 for system prompt and 100 for each turn
        mock_tokenizer.count_tokens.side_effect = lambda text, **kw: len(text.split()) * 10

        system_prompt = "System Instruction"  # length in words = 2 -> 20 tokens
        session = LocalUserSession(
            user_session_id="test_session",
            context=system_prompt,
            system_prompt=system_prompt,
            tokenizer=mock_tokenizer,
            max_model_len=50,
        )

        # turn 1: context grows by 10 tokens
        session.update_context(system_prompt + " Turn1")
        assert session.history == ["Turn1"]
        assert session.context == "System Instruction Turn1"

        # turn 2: context grows by 10 tokens
        session.update_context(session.context + " Turn2")
        assert session.history == ["Turn1", "Turn2"]
        assert session.context == "System Instruction Turn1 Turn2"

        # turn 3: context grows by 10 tokens -> System (20) + 30 = 50 tokens
        session.update_context(session.context + " Turn3")
        assert session.history == ["Turn1", "Turn2", "Turn3"]
        assert session.context == "System Instruction Turn1 Turn2 Turn3"

        # turn 4: context exceeds 50 tokens. First turn ("Turn1") is dropped.
        session.update_context(session.context + " Turn4")
        assert session.history == ["Turn2", "Turn3", "Turn4"]
        assert session.context == "System Instruction Turn2 Turn3 Turn4"


# ---- Open-loop arrival (poisson/constant) helpers and tests ----


def _make_fixed_config(
    num_conversations: int = 3,
    seed: int = 42,
    fixed_turns: int = 3,
    tool_call_latency: float | None = None,
) -> tuple[APIConfig, DataConfig]:
    """Config with a FIXED turn count so arrival/request-count math is exact."""
    api_config = APIConfig(type=APIType.Completion)
    kwargs: dict[str, Any] = dict(
        seed=seed,
        num_conversations=num_conversations,
        shared_system_prompt_len=100,
        turns_per_conversation=Distribution(type="fixed", min=fixed_turns, max=fixed_turns, mean=fixed_turns, std_dev=0),
        input_tokens_per_turn=Distribution(type="fixed", min=20, max=20, mean=20, std_dev=0),
        output_tokens_per_turn=Distribution(type="fixed", min=15, max=15, mean=15, std_dev=0),
    )
    if tool_call_latency is not None:
        kwargs["tool_call_latency_sec"] = Distribution(
            type="fixed", min=tool_call_latency, max=tool_call_latency, mean=tool_call_latency, std_dev=0
        )
    data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=ConversationReplayConfig(**kwargs))
    return api_config, data_config


def _plan(gen: ConversationReplayDataGenerator, num_arrivals: int) -> None:
    """Switch to open-loop and set up a stage of exactly num_arrivals conversations
    (rate=n, duration=1s)."""
    gen.set_closed_loop(False)
    gen.init_stage(StandardLoadStage(rate=float(num_arrivals), duration=1))


def _materialize(gen: ConversationReplayDataGenerator, num_arrivals: int) -> List[Any]:
    """Plan arrivals then materialize every request through the public dispatch path."""
    _plan(gen, num_arrivals)
    # get_data yields _ArrivalLazyData (a LazyLoadInferenceAPIData subclass); narrow it so
    # load_lazy_data's typed parameter is satisfied.
    return [gen.load_lazy_data(cast(_ArrivalLazyData, item)) for item in gen.get_data()]


class TestBlueprintGeneration:
    """Content generation is dispatch-agnostic and unchanged by open-loop arrival."""

    def test_init_creates_correct_number_of_conversations(self) -> None:
        api_config, data_config = _make_config(num_conversations=10)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert len(gen.blueprints) == 10

    def test_deterministic_with_same_seed(self) -> None:
        api_config, data_config = _make_config(seed=123)
        gen1 = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        api_config2, data_config2 = _make_config(seed=123)
        gen2 = ConversationReplayDataGenerator(api_config2, data_config2, _make_mock_tokenizer())

        assert len(gen1.blueprints) == len(gen2.blueprints)
        for bp1, bp2 in zip(gen1.blueprints, gen2.blueprints, strict=True):
            assert bp1.num_turns == bp2.num_turns
            assert bp1.turn_output_lens == bp2.turn_output_lens

    def test_different_seeds_produce_different_results(self) -> None:
        api_config1, data_config1 = _make_config(seed=1)
        gen1 = ConversationReplayDataGenerator(api_config1, data_config1, _make_mock_tokenizer())
        api_config2, data_config2 = _make_config(seed=2)
        gen2 = ConversationReplayDataGenerator(api_config2, data_config2, _make_mock_tokenizer())

        turns1 = [bp.num_turns for bp in gen1.blueprints]
        turns2 = [bp.num_turns for bp in gen2.blueprints]
        assert turns1 != turns2

    def test_requires_tokenizer(self) -> None:
        api_config, data_config = _make_config()
        with pytest.raises(ValueError, match="Tokenizer is required"):
            ConversationReplayDataGenerator(api_config, data_config, None)

    def test_requires_conversation_replay_config(self) -> None:
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(type=DataGenType.ConversationReplay)
        with pytest.raises(ValueError, match="conversation_replay config is required"):
            ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_is_preferred_worker_requested(self) -> None:
        api_config, data_config = _make_config()
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert gen.is_preferred_worker_requested() is True

    def test_tool_call_latency_fixed_distribution(self) -> None:
        """Fixed tool call latency is sampled and stored per turn, and surfaces on the request."""
        api_config, data_config = _make_fixed_config(num_conversations=2, fixed_turns=3, tool_call_latency=5)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        for bp in gen.blueprints:
            assert len(bp.turn_tool_call_latencies) == bp.num_turns
            assert all(lat == 5.0 for lat in bp.turn_tool_call_latencies)

        for req in _materialize(gen, 1):
            assert isinstance(req, _ConversationReplayAPIData)
            assert req.tool_call_latency_sec == 5.0

    def test_tool_call_latency_not_set_gives_zero(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=2, fixed_turns=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        for req in _materialize(gen, 1):
            assert req.tool_call_latency_sec == 0.0

    def test_tool_call_latency_lognormal_distribution(self) -> None:
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=3,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            output_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            tool_call_latency_sec=Distribution(type="lognormal", min=1, max=30, mean=8, std_dev=6),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        for bp in gen.blueprints:
            assert len(bp.turn_tool_call_latencies) == bp.num_turns
            assert not all(lat == bp.turn_tool_call_latencies[0] for lat in bp.turn_tool_call_latencies)
            assert all(1 <= lat <= 30 for lat in bp.turn_tool_call_latencies)


class TestOpenLoopArrival:
    def test_request_count_fixed_turns(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=3, fixed_turns=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        # 5 arrivals over 3 blueprints, each with 3 turns -> 15 requests.
        _plan(gen, 5)
        assert gen.get_request_count() == 15
        assert gen.arrival_turn_counts() == [3, 3, 3, 3, 3]

    def test_arrivals_wrap_blueprints(self) -> None:
        # Variable turns: arrivals beyond num_conversations reuse blueprints round-robin,
        # so request count must equal the summed per-arrival turn counts (with wrap).
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        turns = [bp.num_turns for bp in gen.blueprints]  # 3 blueprints
        _plan(gen, 7)  # arrivals 0..6 -> blueprints 0,1,2,0,1,2,0
        expected = sum(turns[a % 3] for a in range(7))
        assert gen.get_request_count() == expected
        assert gen.arrival_turn_counts() == [turns[a % 3] for a in range(7)]

    def test_zero_arrivals(self) -> None:
        # A stage whose rate*duration rounds to 0 produces no arrivals.
        api_config, data_config = _make_fixed_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        gen.set_closed_loop(False)
        gen.init_stage(StandardLoadStage(rate=0.4, duration=1))  # round(0.4) == 0
        assert gen.get_request_count() == 0
        assert gen.arrival_turn_counts() == []
        assert list(gen.get_data()) == []

    def test_get_data_groups_turns_per_conversation(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=2, fixed_turns=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        _plan(gen, 2)  # 2 arrivals x 3 turns = 6 requests

        items = [cast(_ArrivalLazyData, it) for it in gen.get_data()]
        assert len(items) == 6
        # Each item carries its (arrival, turn) address; turns of an arrival share a worker.
        assert [(it.arrival_index, it.turn_index) for it in items] == [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ]
        assert [it.preferred_worker_id for it in items] == [0, 0, 0, 1, 1, 1]

    def test_fresh_session_per_arrival_and_target_round(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=2, fixed_turns=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        results = _materialize(gen, 2)

        session_ids = [r.user_session_id for r in results]
        assert session_ids == [
            "conv_open_0_0",
            "conv_open_0_0",
            "conv_open_0_0",
            "conv_open_0_1",
            "conv_open_0_1",
            "conv_open_0_1",
        ]
        # Every fresh conversation starts at round 0 and counts up.
        assert [r.target_round for r in results] == [0, 1, 2, 0, 1, 2]
        assert all(isinstance(r, _ConversationReplayAPIData) for r in results)

    def test_prompts_and_max_tokens_from_blueprint(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=1, fixed_turns=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        bp = gen.blueprints[0]
        results = _materialize(gen, 1)
        assert [r.max_tokens for r in results] == bp.turn_output_lens
        assert [r.prompt for r in results] == bp.turn_prompts

    def test_stage_id_in_session_id(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=1, fixed_turns=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        _plan(gen, 1)
        (item,) = [cast(_ArrivalLazyData, it) for it in gen.get_data()][:1]
        item.stage_id = 7
        result = gen.load_lazy_data(item)
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.user_session_id == "conv_open_7_0"

    def test_arrivals_reuse_blueprint_content_with_distinct_sessions(self) -> None:
        # With 1 blueprint and 3 arrivals, every arrival replays the same content
        # but into a distinct session id (no collision).
        api_config, data_config = _make_fixed_config(num_conversations=1, fixed_turns=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        results = _materialize(gen, 3)
        session_ids = {r.user_session_id for r in results}
        assert session_ids == {"conv_open_0_0", "conv_open_0_1", "conv_open_0_2"}


class TestClosedLoop:
    """Closed-loop (concurrent) dispatch: infinite round-robin over slots with recycling.

    The datagen defaults to closed-loop, so these need no set_closed_loop call.
    """

    def test_get_data_yields_lazy_load_with_preferred_worker(self) -> None:
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        data_iter = gen.get_data()
        items = [next(data_iter) for _ in range(9)]

        # First 3 items should cycle through conversations 0, 1, 2
        assert all(isinstance(item, LazyLoadInferenceAPIData) for item in items)
        assert items[0].preferred_worker_id == 0
        assert items[1].preferred_worker_id == 1
        assert items[2].preferred_worker_id == 2
        # Second round
        assert items[3].preferred_worker_id == 0

    def test_load_lazy_data_returns_user_session_data(self) -> None:
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)

        assert isinstance(result, _ConversationReplayAPIData)
        assert result.user_session == gen.user_sessions[0]
        assert result.target_round == 0

    def test_turn_recycling(self) -> None:
        """When data_index exceeds total turns, it wraps around."""
        api_config, data_config = _make_config(num_conversations=2, turns_min=3, turns_max=3, turns_mean=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        # Conversation 0 has 3 turns. data_index=0 -> round 0, turn 0
        # data_index=6 -> conv 0, round 3, turn 0 (recycled)
        lazy = LazyLoadInferenceAPIData(data_index=6, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.target_round == 3  # 6 // 2 = 3

    def test_user_session_ids(self) -> None:
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        ids = [s.user_session_id for s in gen.user_sessions]
        assert ids == ["conv_0", "conv_1", "conv_2"]

    def test_slot_reset_on_new_conversation(self) -> None:
        # On turn_idx==0 with round_num>0 the slot resets to a fresh session id.
        api_config, data_config = _make_config(num_conversations=2, turns_min=3, turns_max=3, turns_mean=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        # data_index=4 -> conv 0, round_num=2, turn_idx=2 (still convo 0); data_index=6 -> round 3,
        # turn 0, convo 1 -> reset to slot_0_convo_1.
        result = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=6, preferred_worker_id=0))
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.user_session_id == "slot_0_convo_1"

    def test_reprime_after_clear_instances(self) -> None:
        # After the LoadGenerator clears the session registry between stages, load_lazy_data
        # must re-create the slot's conv_{i} session.
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        LocalUserSession.clear_instances()
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=1))
        assert "conv_1" in LocalUserSession._instances

    def test_init_stage_noop_does_not_crash_on_concurrent_stage(self) -> None:
        # mp_run calls init_stage for every stage; closed-loop must no-op on ConcurrentLoadStage
        # (not a StandardLoadStage) rather than assert-crash.
        from inference_perf.config import ConcurrentLoadStage

        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        gen.init_stage(ConcurrentLoadStage(num_requests=4, concurrency_level=2))  # must not raise
        assert gen.get_request_count() == 0


def _config_without_num_conversations(fixed_turns: int = 2, seed: int = 42) -> tuple[APIConfig, DataConfig]:
    """conversation_replay config that leaves num_conversations unset (auto-size)."""
    api_config = APIConfig(type=APIType.Completion)
    cr_config = ConversationReplayConfig(
        seed=seed,
        shared_system_prompt_len=100,
        turns_per_conversation=Distribution(type="fixed", min=fixed_turns, max=fixed_turns, mean=fixed_turns, std_dev=0),
        input_tokens_per_turn=Distribution(type="fixed", min=20, max=20, mean=20, std_dev=0),
        output_tokens_per_turn=Distribution(type="fixed", min=15, max=15, mean=15, std_dev=0),
    )
    assert cr_config.num_conversations is None  # unset by default now
    data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)
    return api_config, data_config


class TestPoolSizing:
    def test_config_value_used(self) -> None:
        # main.py writes the resolved count onto the config; the datagen reads it.
        api_config, data_config = _make_fixed_config(num_conversations=4)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert len(gen.blueprints) == 4

    def test_raises_when_unset(self) -> None:
        # Unset config value and no main.py write-back -> fail loudly, no implicit default.
        api_config, data_config = _config_without_num_conversations()
        with pytest.raises(AssertionError, match="num_conversations must be resolved"):
            ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_auto_sized_pool_makes_every_arrival_unique(self) -> None:
        # Emulate main.py's write-back: set the resolved count on the config, then build.
        # When the pool equals the arrival count, no blueprint index repeats, so every
        # arrival maps to its own blueprint (a % n == a for a < n).
        api_config, data_config = _config_without_num_conversations(fixed_turns=2)
        arrivals = 6
        assert data_config.conversation_replay is not None
        data_config.conversation_replay.num_conversations = arrivals
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        results = _materialize(gen, arrivals)
        # Distinct sessions AND distinct underlying blueprint prompts (no reuse).
        session_ids = {r.user_session_id for r in results}
        assert len(session_ids) == arrivals
        first_turn_prompts = [gen.blueprints[a].turn_prompts[0] for a in range(arrivals)]
        assert len(set(first_turn_prompts)) == len(first_turn_prompts)  # all blueprints distinct


class TestArrivalRepeatLoadTimer:
    def test_repeats_each_tick_per_turn_count(self) -> None:
        # Base timer yields one tick per arrival; wrapper repeats per turn count.
        base = ConstantLoadTimer(rate=100.0, duration=10.0)  # plenty of ticks available
        timer = ArrivalRepeatLoadTimer(base, turn_counts=[2, 3])
        ticks = list(timer.start_timer(initial=0.0))
        assert len(ticks) == 5
        assert ticks[0] == ticks[1]  # arrival 0, 2 turns
        assert ticks[2] == ticks[3] == ticks[4]  # arrival 1, 3 turns
        assert ticks[0] < ticks[2]  # arrivals ordered in time


class TestSystemPromptGeneration:
    """System-prompt generation (shared prefix per stage + dynamic suffix) under open-loop."""

    def _det_tok(self) -> MagicMock:
        mock_tok = MagicMock()
        hf_tok = MagicMock()
        hf_tok.vocab_size = 32000
        hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
        mock_tok.get_tokenizer.return_value = hf_tok
        return mock_tok

    def test_shared_prompt_reused_across_arrivals_in_stage(self) -> None:
        # Two arrivals from the SAME blueprint share the stage's shared system prompt.
        api_config, data_config = _make_fixed_config(num_conversations=1, fixed_turns=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, self._det_tok())
        _materialize(gen, 2)
        c0 = LocalUserSession._instances["conv_open_0_0"].context
        c1 = LocalUserSession._instances["conv_open_0_1"].context
        assert c0 == c1  # same blueprint, same stage prompt

    def test_shared_prompt_differs_across_stages(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=1, fixed_turns=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, self._det_tok())

        _plan(gen, 1)
        item = cast(_ArrivalLazyData, next(iter(gen.get_data())))
        item.stage_id = 0
        gen.load_lazy_data(item)
        ctx_stage0 = LocalUserSession._instances["conv_open_0_0"].context

        LocalUserSession.clear_instances()
        item2 = cast(_ArrivalLazyData, next(iter(gen.get_data())))
        item2.stage_id = 1
        gen.load_lazy_data(item2)
        ctx_stage1 = LocalUserSession._instances["conv_open_1_0"].context

        assert ctx_stage0 != ctx_stage1

    def test_shared_prompt_generated_once_per_stage(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=2, fixed_turns=1)
        mock_tok = _make_mock_tokenizer()
        decode_count = 0

        def counting_decode(ids: Any, **kwargs: Any) -> str:
            nonlocal decode_count
            decode_count += 1
            return f"decoded_{ids}"

        gen = ConversationReplayDataGenerator(api_config, data_config, mock_tok)
        mock_tok.get_tokenizer().decode.side_effect = counting_decode
        decode_count = 0

        # Two arrivals from two different blueprints, one turn each, same stage.
        # The shared prefix is generated once and cached; the dynamic suffix is empty
        # (no dynamic_system_prompt_len configured), so only the shared prefix decodes.
        _plan(gen, 2)
        for item in (cast(_ArrivalLazyData, it) for it in gen.get_data()):
            item.stage_id = 5
            gen.load_lazy_data(item)
        assert decode_count == 1

    def test_reproducible_across_runs_same_seed(self) -> None:
        api_config, data_config = _make_fixed_config(num_conversations=2, fixed_turns=1)

        def run() -> str:
            LocalUserSession.clear_instances()
            gen = ConversationReplayDataGenerator(api_config, data_config, self._det_tok())
            _plan(gen, 1)
            item = cast(_ArrivalLazyData, next(iter(gen.get_data())))
            item.stage_id = 0
            gen.load_lazy_data(item)
            return LocalUserSession._instances["conv_open_0_0"].context

        assert run() == run()
