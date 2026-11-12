"""Resource caps: keep SmartAMR inside a share of the machine it runs on.

The embedding step is the greedy one. ESM-C holds the model plus one protein's
activations on the GPU, and the feature builder holds every embedding in RAM at
once, so an unbounded run either triggers the OOM killer or takes the desktop
down with it. This module puts one number in one place -- the fraction of the
machine SmartAMR may use -- and applies it to the three things that actually
grow: GPU memory, host memory, and thread pools.

The default is half the machine (``SMARTAMR_RESOURCE_FRACTION=0.5``).

Three caps, three mechanisms:

``GPU``
    ``torch.cuda.set_per_process_memory_fraction`` makes the CUDA caching
    allocator refuse an allocation that would push this process past the share.
    The process raises ``torch.OutOfMemoryError`` instead of the driver evicting
    somebody else's context. The fraction is of the *total* card, so a card that
    is already busy is measured honestly: see :func:`gpu_budget`.

``RAM``
    ``RLIMIT_AS`` would be the hard way to do it, but it counts reserved address
    space rather than resident pages, and both PyTorch and the CUDA driver
    reserve enormous unbacked ranges -- the limit fires on a healthy process.
    So memory is enforced cooperatively: :meth:`ResourceLimits.memory_guard`
    reads the real RSS and raises before a step balloons past the share.

``Threads``
    BLAS and OpenMP default to one thread per core and oversubscribe badly
    under scikit-learn's own parallelism. Capped to the same share of cores.

Nothing here is a security boundary; a determined allocation can still exceed
the share. It is a guard rail against this project's own appetite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

#: Environment variable holding the fraction of the machine SmartAMR may use.
FRACTION_ENV_VAR = "SMARTAMR_RESOURCE_FRACTION"

#: Half the machine, as the project's stated budget.
DEFAULT_FRACTION = 0.5

#: Below this many bytes a GPU share is not worth using; fall back to CPU.
MIN_USABLE_GPU_BYTES = 1 << 30  # 1 GiB

_BYTES_PER_GIB = float(1 << 30)


def _env_fraction() -> float:
    """The configured fraction, clamped to a sane open interval."""
    raw = os.environ.get(FRACTION_ENV_VAR)
    if not raw:
        return DEFAULT_FRACTION
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_FRACTION
    # 0 would forbid every allocation and >1 is not a share of anything.
    return min(max(value, 0.05), 1.0)


def _total_memory_bytes() -> int:
    """Total physical RAM, read from the kernel rather than guessed."""
    # os.sysconf is present on Linux and macOS and needs no third-party import,
    # which matters because psutil is not a hard dependency of this package.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (OSError, ValueError, AttributeError):
        pass
    try:  # pragma: no cover - only on exotic platforms
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        return 8 * (1 << 30)  # a conservative guess, better than crashing


def _process_rss_bytes() -> int:
    """Resident set size of this process, or 0 when it cannot be read."""
    # /proc/self/statm is two syscalls and no dependency. Field 2 is resident
    # pages. psutil is only consulted if /proc is absent (macOS, containers).
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return 0


def _cpu_count() -> int:
    """Cores available to this process, honouring CPU affinity when set."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def gpu_budget(fraction: float) -> tuple[int, int]:
    """``(budget_bytes, total_bytes)`` for this process on GPU 0.

    The budget is ``fraction`` of the card's **total** memory. "Cap the GPU at
    50%" is a statement about the card, so that is what is enforced, and the
    number does not move when another process starts or stops.

    An earlier version subtracted whatever other processes held at the moment
    the caps were resolved. That sounded more considerate but behaved badly:
    the budget was computed once at start-up, so a card that happened to be
    busy then left this process permanently squeezed -- here it produced a
    1.08 GiB ceiling that OOM-failed every long protein, and the ceiling stayed
    after the other process exited and freed 3 GiB. Current free memory is
    still reported by :func:`gpu_free_bytes` so a caller can warn, but it no
    longer shrinks the cap.

    Returns ``(0, 0)`` when there is no usable CUDA device.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            return 0, 0
        total_bytes = int(torch.cuda.mem_get_info()[1])
    except Exception:  # noqa: BLE001 - no torch, no driver, no device
        return 0, 0

    return int(fraction * total_bytes), total_bytes


def gpu_free_bytes() -> int:
    """Free memory on GPU 0 right now, or 0 when there is no device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001
        return 0


@dataclass(frozen=True)
class ResourceLimits:
    """The share of this machine SmartAMR is allowed to use.

    Build it with :meth:`from_env` and call :meth:`apply` once, early, before
    torch allocates anything.
    """

    fraction: float = DEFAULT_FRACTION
    memory_budget_bytes: int = 0
    total_memory_bytes: int = 0
    gpu_budget_bytes: int = 0
    total_gpu_bytes: int = 0
    threads: int = 1
    total_threads: int = 1

    @classmethod
    def from_env(cls, fraction: float | None = None) -> "ResourceLimits":
        """Resolve the caps from the machine and the environment."""
        share = _env_fraction() if fraction is None else min(max(fraction, 0.05), 1.0)
        total_memory = _total_memory_bytes()
        total_threads = _cpu_count()
        gpu_bytes, total_gpu = gpu_budget(share)
        return cls(
            fraction=share,
            memory_budget_bytes=int(share * total_memory),
            total_memory_bytes=total_memory,
            gpu_budget_bytes=gpu_bytes,
            total_gpu_bytes=total_gpu,
            threads=max(1, int(share * total_threads)),
            total_threads=total_threads,
        )

    # --- properties used for reporting ---------------------------------
    @property
    def memory_budget_gib(self) -> float:
        return self.memory_budget_bytes / _BYTES_PER_GIB

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory_bytes / _BYTES_PER_GIB

    @property
    def gpu_budget_gib(self) -> float:
        return self.gpu_budget_bytes / _BYTES_PER_GIB

    @property
    def total_gpu_gib(self) -> float:
        return self.total_gpu_bytes / _BYTES_PER_GIB

    @property
    def gpu_usable(self) -> bool:
        """True when the GPU share is big enough to be worth using."""
        return self.gpu_budget_bytes >= MIN_USABLE_GPU_BYTES

    def device(self, prefer: str = "cuda") -> str:
        """``"cuda"`` when a usable share of the card exists, else ``"cpu"``."""
        if prefer.startswith("cuda") and self.gpu_usable:
            return prefer
        return "cpu"

    # --- enforcement ---------------------------------------------------
    def apply_thread_limits(self) -> None:
        """Cap BLAS/OpenMP thread pools to the share of the cores.

        These are read by the numeric libraries when they initialise, so this
        has to run before numpy, torch or sklearn import their backends. Values
        already set by the user are left alone.
        """

        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(variable, str(self.threads))

    def apply_gpu_limit(self) -> bool:
        """Cap this process's CUDA allocations. True when a cap was installed."""
        if not self.gpu_usable or self.total_gpu_bytes <= 0:
            return False
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            # The API takes a fraction of the *total* card, so convert the
            # byte budget back into that unit.
            share_of_card = min(self.gpu_budget_bytes / self.total_gpu_bytes, 1.0)
            torch.cuda.set_per_process_memory_fraction(share_of_card, 0)
            return True
        except Exception:  # noqa: BLE001 - old torch, no driver
            return False

    def apply_allocator_hints(self) -> None:
        """Ask the CUDA allocator for expandable segments.

        A capped process fragments its own pool: variable-length proteins make
        allocations of many different sizes, and a cached block of the wrong
        size is dead weight against the ceiling. Expandable segments let the
        allocator grow one region instead of keeping a size-bucketed pool, which
        is what turns "126 MiB reserved but unallocated" into usable headroom.
        Must be set before the first CUDA allocation.
        """

        existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in existing:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"{existing},expandable_segments:True".lstrip(",")
            )

    def apply(self) -> "ResourceLimits":
        """Install every cap that can be installed. Returns ``self``."""
        self.apply_thread_limits()
        self.apply_allocator_hints()
        self.apply_gpu_limit()
        return self

    def memory_headroom_bytes(self) -> int:
        """Bytes of the RAM budget still unused by this process."""
        return max(self.memory_budget_bytes - _process_rss_bytes(), 0)

    def memory_exceeded(self) -> bool:
        """True when this process is already over its RAM share."""
        return _process_rss_bytes() > self.memory_budget_bytes

    def memory_guard(self, step: str) -> None:
        """Raise :class:`MemoryBudgetExceeded` if the RAM share is spent.

        Call this between units of work -- per phage, per chunk -- so a run
        that is heading for the OOM killer stops with an explanatory error and
        whatever it has already written stays on disk.
        """

        resident = _process_rss_bytes()
        if resident > self.memory_budget_bytes:
            raise MemoryBudgetExceeded(
                f"{step}: using {resident / _BYTES_PER_GIB:.1f} GiB, over the "
                f"{self.memory_budget_gib:.1f} GiB budget "
                f"({self.fraction:.0%} of {self.total_memory_gib:.1f} GiB). "
                f"Raise {FRACTION_ENV_VAR} or process fewer items at a time."
            )

    def summary(self) -> str:
        """One-line, human-readable statement of the caps."""
        if self.total_gpu_bytes > 0:
            gpu = f"{self.gpu_budget_gib:.1f}/{self.total_gpu_gib:.1f} GiB"
            if not self.gpu_usable:
                gpu += " (too small, using CPU)"
        else:
            gpu = "none detected"
        return (
            f"Resource cap {self.fraction:.0%} -> "
            f"RAM {self.memory_budget_gib:.1f}/{self.total_memory_gib:.1f} GiB, "
            f"GPU {gpu}, "
            f"threads {self.threads}/{self.total_threads}"
        )


class MemoryBudgetExceeded(MemoryError):
    """Raised when a step would push the process past its RAM share."""


def limits(fraction: float | None = None) -> ResourceLimits:
    """Resolve and install the caps in one call."""
    return ResourceLimits.from_env(fraction).apply()


def chunked(items: list, size: int) -> Iterator[list]:
    """Yield ``items`` in lists of at most ``size`` -- the unit of memory work."""
    if size < 1:
        raise ValueError("chunk size must be at least 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]
