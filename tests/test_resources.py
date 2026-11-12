"""Tests for the resource caps.

These never touch a GPU: the CUDA paths degrade to "no device" without torch,
and that fallback is itself worth asserting.
"""

from __future__ import annotations

import pytest

from SmartAMR.resources import (
    DEFAULT_FRACTION,
    FRACTION_ENV_VAR,
    MIN_USABLE_GPU_BYTES,
    MemoryBudgetExceeded,
    ResourceLimits,
    chunked,
)


def test_default_fraction_is_half():
    limits = ResourceLimits.from_env()
    assert limits.fraction == DEFAULT_FRACTION == 0.5


def test_fraction_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV_VAR, "0.25")
    assert ResourceLimits.from_env().fraction == 0.25


def test_an_unparseable_fraction_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV_VAR, "half of it")
    assert ResourceLimits.from_env().fraction == DEFAULT_FRACTION


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0.05), ("-3", 0.05), ("2.5", 1.0), ("1", 1.0)],
)
def test_the_fraction_is_clamped_to_a_usable_range(monkeypatch, value, expected):
    # 0 would forbid every allocation and >1 is not a share of anything.
    monkeypatch.setenv(FRACTION_ENV_VAR, value)
    assert ResourceLimits.from_env().fraction == expected


def test_the_budget_is_the_fraction_of_total_memory():
    limits = ResourceLimits.from_env(0.5)
    assert limits.total_memory_bytes > 0
    assert limits.memory_budget_bytes == int(0.5 * limits.total_memory_bytes)


def test_threads_are_a_share_of_the_cores_but_never_zero():
    limits = ResourceLimits.from_env(0.05)
    assert limits.threads >= 1
    assert limits.threads <= limits.total_threads


def test_thread_limits_reach_the_numeric_libraries(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    limits = ResourceLimits.from_env(0.5)
    limits.apply_thread_limits()

    import os

    assert os.environ["OMP_NUM_THREADS"] == str(limits.threads)


def test_thread_limits_do_not_override_an_explicit_setting(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    ResourceLimits.from_env(0.5).apply_thread_limits()

    import os

    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_the_allocator_hint_is_set_once(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    limits = ResourceLimits.from_env()
    limits.apply_allocator_hints()
    limits.apply_allocator_hints()

    import os

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"].count("expandable_segments") == 1


def test_memory_guard_raises_once_the_budget_is_spent():
    # A one-byte budget is already exceeded by the interpreter itself.
    limits = ResourceLimits(fraction=0.5, memory_budget_bytes=1, total_memory_bytes=2)
    with pytest.raises(MemoryBudgetExceeded) as error:
        limits.memory_guard("a test step")

    message = str(error.value)
    assert "a test step" in message
    # The message has to say how to proceed, not just that it stopped.
    assert FRACTION_ENV_VAR in message


def test_memory_guard_is_silent_inside_the_budget():
    limits = ResourceLimits(
        fraction=0.5, memory_budget_bytes=1 << 62, total_memory_bytes=1 << 63
    )
    limits.memory_guard("plenty of room")  # must not raise


def test_a_tiny_gpu_share_is_not_used():
    limits = ResourceLimits(gpu_budget_bytes=MIN_USABLE_GPU_BYTES - 1, total_gpu_bytes=1 << 33)
    assert not limits.gpu_usable
    assert limits.device("cuda") == "cpu"


def test_a_real_gpu_share_is_used():
    limits = ResourceLimits(gpu_budget_bytes=4 << 30, total_gpu_bytes=8 << 30)
    assert limits.gpu_usable
    assert limits.device("cuda") == "cuda"


def test_cpu_is_honoured_when_asked_for():
    limits = ResourceLimits(gpu_budget_bytes=4 << 30, total_gpu_bytes=8 << 30)
    assert limits.device("cpu") == "cpu"


def test_applying_the_gpu_limit_without_a_device_is_a_no_op():
    assert ResourceLimits(gpu_budget_bytes=0, total_gpu_bytes=0).apply_gpu_limit() is False


def test_the_summary_names_every_cap():
    summary = ResourceLimits.from_env(0.5).summary()
    assert "50%" in summary
    assert "RAM" in summary
    assert "GPU" in summary
    assert "threads" in summary


def test_chunked_splits_without_losing_items():
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunked([], 3)) == []


def test_chunked_rejects_a_zero_size():
    with pytest.raises(ValueError):
        list(chunked([1, 2], 0))
