"""Bounded chart data from detected beats; no model-derived signal measurements."""
import numpy as np
from reporting.schemas import ProcessedECGSignal, SignalAnalytics


def build_signal_analytics(signal: ProcessedECGSignal, max_points: int = 1500) -> SignalAnalytics:
    if max_points < 2:
        raise ValueError("Charts require at least two points.")
    rr = np.asarray(signal.rr_intervals, dtype=float)
    times = np.asarray(signal.r_peaks[1:], dtype=float) / signal.sampling_rate
    if len(rr) != len(times) or len(rr) < 2 or not np.isfinite(rr).all() or np.any(rr <= 0):
        raise ValueError("Chart intervals must be finite, positive and aligned with detected beats.")
    indices = np.linspace(0, len(rr) - 1, min(len(rr), max_points), dtype=int)
    pair_indices = np.linspace(0, len(rr) - 2, min(len(rr) - 1, max_points), dtype=int)
    heart_rate = 60000.0 / rr
    return SignalAnalytics(
        time_seconds=times[indices].tolist(), rr_intervals_ms=rr[indices].tolist(),
        heart_rate_bpm=heart_rate[indices].tolist(),
        consecutive_rr_ms=np.column_stack((rr[pair_indices], rr[pair_indices + 1])).tolist(),
        total_intervals=len(rr), mean_rr_ms=float(np.mean(rr)),
        mean_heart_rate_bpm=float(60000.0 / np.mean(rr)),
        min_heart_rate_bpm=float(np.min(heart_rate)), max_heart_rate_bpm=float(np.max(heart_rate)),
    )
