"""
Per-step and per-epoch efficiency tracking for training runs.

``EfficiencyTracker`` accumulates wall-clock timings and GPU memory stats
during a training epoch.  It is intentionally decoupled from the training
loop so that the same metrics can be collected for any task.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch


class EfficiencyTracker:
    """Accumulate per-step efficiency measurements across one epoch.

    Usage::

        tracker = EfficiencyTracker(device="cuda", warmup_steps=5)
        tracker.start_epoch()

        for step, batch in enumerate(loader):
            tracker.record_batch_start()          # start of data-loading gap
            # ... move batch to device ...
            tracker.record_compute_start()        # before forward/backward
            # ... forward / backward / step ...
            tracker.record_compute_end(n_tokens=input_ids.numel())

        summary = tracker.end_epoch()
        # summary contains: avg_step_ms_e2e, tokens_per_sec_e2e, etc.

    Attributes:
        warmup_steps: Number of steps skipped at the start of each epoch
            before measurements begin (avoids JIT-compilation overhead).
    """

    def __init__(
        self, device: Optional[str] = None, warmup_steps: int = 5
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup_steps = warmup_steps
        self._reset()

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._step = 0
        self._data_times: List[float] = []
        self._compute_times: List[float] = []
        self._step_times: List[float] = []
        self._total_tokens = 0
        self._total_compute_time = 0.0
        self._total_step_time = 0.0
        self._prev_batch_yield = 0.0
        self._compute_start = 0.0
        self._epoch_start = 0.0

    # ------------------------------------------------------------------
    # Per-step recording API
    # ------------------------------------------------------------------

    def start_epoch(self) -> None:
        """Call once at the start of an epoch."""
        self._reset()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._epoch_start = time.perf_counter()
        self._prev_batch_yield = self._epoch_start

    def record_batch_start(self) -> None:
        """Call immediately after receiving a batch from the DataLoader.

        Measures the data-loading gap (time between this call and the
        previous ``record_compute_end``).
        """
        self._data_time = time.perf_counter() - self._prev_batch_yield

    def record_compute_start(self) -> None:
        """Call just before the forward + backward pass."""
        if self.device == "cuda":
            torch.cuda.synchronize()
        self._compute_start = time.perf_counter()

    def record_compute_end(self, n_tokens: int) -> None:
        """Call immediately after the optimizer step.

        Args:
            n_tokens: Number of token positions in this batch (including
                padding — ``input_ids.numel()``).
        """
        if self.device == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        compute_time = now - self._compute_start
        step_time = self._data_time + compute_time
        self._prev_batch_yield = now

        if self._step >= self.warmup_steps:
            self._data_times.append(self._data_time)
            self._compute_times.append(compute_time)
            self._step_times.append(step_time)
            self._total_tokens += n_tokens
            self._total_compute_time += compute_time
            self._total_step_time += step_time

        self._step += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def end_epoch(self) -> Dict[str, float]:
        """Finalise the epoch and return a summary dict.

        Returns:
            Dict with keys:
            - ``avg_step_ms_e2e``       — mean end-to-end step time (ms)
            - ``tokens_per_sec_e2e``    — throughput including data loading
            - ``avg_step_ms_compute``   — mean compute-only step time (ms)
            - ``tokens_per_sec_compute``— throughput for forward+backward
            - ``avg_data_ms``           — mean data-loading time (ms)
            - ``avg_compute_ms``        — mean compute time (ms)
            - ``epoch_wall_s``          — total epoch wall-clock time (s)
            - ``peak_mem_alloc_gb``     — peak GPU memory allocated (GB)
            - ``peak_mem_reserved_gb``  — peak GPU memory reserved (GB)
        """
        epoch_wall = time.perf_counter() - self._epoch_start
        nan = float("nan")

        if self._step_times:
            avg_step_e2e = 1000.0 * float(np.mean(self._step_times))
            avg_step_compute = 1000.0 * float(np.mean(self._compute_times))
            avg_data_ms = 1000.0 * float(np.mean(self._data_times))
            avg_compute_ms = 1000.0 * float(np.mean(self._compute_times))
            tok_e2e = (
                self._total_tokens / self._total_step_time
                if self._total_step_time > 0
                else nan
            )
            tok_compute = (
                self._total_tokens / self._total_compute_time
                if self._total_compute_time > 0
                else nan
            )
        else:
            avg_step_e2e = avg_step_compute = avg_data_ms = avg_compute_ms = nan
            tok_e2e = tok_compute = nan

        if self.device == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        else:
            peak_alloc = peak_reserved = nan

        return {
            "avg_step_ms_e2e": avg_step_e2e,
            "tokens_per_sec_e2e": tok_e2e,
            "avg_step_ms_compute": avg_step_compute,
            "tokens_per_sec_compute": tok_compute,
            "avg_data_ms": avg_data_ms,
            "avg_compute_ms": avg_compute_ms,
            "epoch_wall_s": epoch_wall,
            "peak_mem_alloc_gb": peak_alloc,
            "peak_mem_reserved_gb": peak_reserved,
        }
