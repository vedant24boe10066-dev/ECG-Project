import numpy as np
import pytest
from reporting.schemas import ProcessedECGSignal
from reporting.signal_analytics import build_signal_analytics


def signal(peaks):
    peaks = np.asarray(peaks)
    return ProcessedECGSignal(filtered_signal=np.zeros(peaks[-1] + 1), r_peaks=peaks,
                              rr_intervals=np.diff(peaks) * 4., sampling_rate=250.)


def test_chart_values_match_detected_beats():
    result = build_signal_analytics(signal([100, 350, 650, 850]))
    assert result.time_seconds == [1.4, 2.6, 3.4]
    assert result.rr_intervals_ms == [1000., 1200., 800.]
    assert result.heart_rate_bpm == [60., 50., 75.]
    assert result.consecutive_rr_ms == [[1000., 1200.], [1200., 800.]]
    assert result.mean_heart_rate_bpm == 60.
    assert result.min_heart_rate_bpm == 50.
    assert result.max_heart_rate_bpm == 75.


def test_bounded_charts_keep_original_adjacent_pairs_and_full_summary():
    gaps = np.arange(200, 2200)
    source = signal(np.concatenate(([0], np.cumsum(gaps))))
    result = build_signal_analytics(source, max_points=10)
    assert len(result.time_seconds) == len(result.consecutive_rr_ms) == 10
    assert result.total_intervals == 2000
    assert result.time_seconds[-1] == source.r_peaks[-1] / 250
    assert result.mean_rr_ms == np.mean(source.rr_intervals)
    for first, second in result.consecutive_rr_ms:
        assert second - first == 4.  # Adjacent original beats, not adjacent sampled points.


def test_invalid_intervals_rejected():
    source = signal([100, 350, 650, 850])
    source.rr_intervals[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        build_signal_analytics(source)
