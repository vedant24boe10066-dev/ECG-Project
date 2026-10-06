"""Regression tests for input validation, model integrity, and API reports."""
import asyncio
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile

from features.hrv_extractor import NeuroKitHRVExtractor
from ingestion.csv_ingestor import CSVECGIngestor
from models.classifiers import RandomForestPredictor
from preprocessing.filters import NeuroKitPreprocessor
from reporting.schemas import ProcessedECGSignal, RawECGSignal

ROOT = Path(__file__).resolve().parents[1]


def test_demo_is_reproducible_and_missing_features_are_explicit():
    features = {"SDNN": 12.0, "RMSSD": None}
    first = RandomForestPredictor().predict(features)
    second = RandomForestPredictor().predict(features)
    assert first.is_demo and second.is_demo
    assert first.stress_probability == second.stress_probability
    assert features["RMSSD"] is None


def test_training_save_load_feature_order_and_no_silent_retraining(tmp_path):
    X = pd.DataFrame({"SDNN": [1, 2, 10, 20], "RMSSD": [5, 6, 30, 40]})
    predictor = RandomForestPredictor()
    predictor.fit(X, np.array([0, 0, 1, 1]))
    expected = predictor.predict({"SDNN": 10, "RMSSD": 30})
    path = tmp_path / "model.joblib"
    predictor.save_model(str(path))
    loaded = RandomForestPredictor(model_path=str(path))
    actual = loaded.predict({"RMSSD": 30, "SDNN": 10})
    assert not actual.is_demo
    assert actual.stress_probability == expected.stress_probability
    model = loaded.get_model()
    with pytest.raises(ValueError, match="training feature names"):
        loaded.predict({"SDNN": 10, "other": 30})
    with pytest.raises(ValueError, match="unavailable HRV"):
        loaded.predict({"SDNN": 10, "RMSSD": None})
    assert loaded.get_model() is model
    with pytest.raises(FileNotFoundError):
        RandomForestPredictor(model_path=str(tmp_path / "missing.joblib"))


def test_demo_flag_survives_saved_model(tmp_path):
    predictor = RandomForestPredictor()
    predictor.predict({"SDNN": 12})
    path = tmp_path / "demo.joblib"
    predictor.save_model(str(path))
    assert RandomForestPredictor(model_path=str(path)).predict({"SDNN": 12}).is_demo


@pytest.mark.parametrize("labels", [[0, 0], [1, 1], [1, 2]])
def test_training_requires_both_binary_classes(labels):
    with pytest.raises(ValueError, match="both 0"):
        RandomForestPredictor().fit(pd.DataFrame({"SDNN": [1, 2]}), np.array(labels))


@pytest.mark.parametrize("filename", ["ECG (EO, AC1, AC2).csv", "ecg.csv"])
def test_incompatible_supplied_csv_is_rejected(filename):
    with pytest.raises(ValueError, match="CSV must have"):
        CSVECGIngestor(250).ingest(str(ROOT / "sample_csv" / filename))


def test_explicit_signal_column_with_time_column(tmp_path):
    path = tmp_path / "signal.csv"
    pd.DataFrame({"time": np.arange(2500), "ECG": np.sin(np.arange(2500))}).to_csv(path, index=False)
    raw = CSVECGIngestor(250).ingest(str(path))
    assert np.allclose(raw.signal_data, np.sin(np.arange(2500)))


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), 250.5])
def test_invalid_sampling_rates(rate):
    with pytest.raises(ValueError):
        CSVECGIngestor(rate).ingest(str(ROOT / "sample_ecg.csv"))


@pytest.mark.parametrize("signal", [np.array([1.0, 2.0]), np.full(2500, np.nan), np.full(2500, np.inf), np.ones((2500, 2))])
def test_invalid_signals(signal):
    with pytest.raises(ValueError):
        RawECGSignal(signal_data=signal, sampling_rate=250, duration_seconds=10)


def test_constant_signal_and_insufficient_peaks():
    raw = RawECGSignal(signal_data=np.ones(2500), sampling_rate=250, duration_seconds=10)
    with pytest.raises(ValueError, match="constant"):
        NeuroKitPreprocessor().preprocess(raw)
    processed = ProcessedECGSignal(filtered_signal=np.zeros(2500), r_peaks=np.array([10, 100]), rr_intervals=np.array([360.]), sampling_rate=250)
    with pytest.raises(ValueError, match="at least 10"):
        NeuroKitHRVExtractor().extract_features(processed)


def test_unavailable_metrics_are_null_and_zero_is_preserved(monkeypatch):
    import features.hrv_extractor as module
    monkeypatch.setattr(module.nk, "hrv", lambda *args, **kwargs: pd.DataFrame({"HRV_SDNN": [np.nan], "HRV_RMSSD": [0.], "HRV_LF": [np.inf]}))
    processed = ProcessedECGSignal(filtered_signal=np.zeros(2500), r_peaks=np.arange(100, 2100, 200), rr_intervals=np.ones(9) * 800, sampling_rate=250)
    metrics = NeuroKitHRVExtractor().extract_features(processed)
    assert metrics.time_domain["SDNN"] is None
    assert metrics.time_domain["RMSSD"] == 0.
    assert metrics.frequency_domain["LF"] is None
    assert metrics.warnings
    assert 'NaN' not in metrics.model_dump_json() and 'Infinity' not in metrics.model_dump_json()


@pytest.mark.parametrize("filename,expected", [("sample_ecg.csv", "complete"), ("sample_csv/ECG (EO, AC1, AC2).csv", "error")])
def test_api_stream_reports_demo_or_clear_input_error(tmp_path, monkeypatch, filename, expected):
    import api
    payload = (ROOT / filename).read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api, "predictor", RandomForestPredictor())

    async def consume():
        upload = UploadFile(filename=Path(filename).name, file=io.BytesIO(payload))
        response = await api.analyze_ecg(files=[upload], sampling_rate=250)
        return [json.loads(line) async for line in response.body_iterator]

    messages = asyncio.run(consume())
    assert messages[-1]["status"] == expected
    assert not list((tmp_path / "data/raw").iterdir())
    if expected == "complete":
        report = messages[-1]["data"]
        assert report["prediction"]["is_demo"]
        assert report["pipeline_metadata"]["r_peaks_detected"] == 34
        assert len(report["prediction"]["top_shap_features"]) == 5
        assert report["key_hrv_summary"]["LF_HF_Ratio"] is None
        assert report["pipeline_metadata"]["warnings"]
    else:
        assert "CSV must have" in messages[-1]["message"]

@pytest.mark.parametrize("filename,expected", [
    ("01_baseline_70bpm.csv", "No-Stress"),
    ("02_slower_60bpm.csv", "No-Stress"),
    ("03_faster_90bpm.csv", "Stress"),
    ("04_faster_105bpm.csv", "Stress"),
    ("05_low_variability_75bpm.csv", "Stress"),
    ("06_higher_variability_75bpm.csv", "No-Stress"),
    ("07_light_noise_70bpm.csv", "No-Stress"),
    ("08_baseline_wander_70bpm.csv", "No-Stress"),
    ("09_longer_80bpm.csv", "Stress"),
    ("10_longer_variable_65bpm.csv", "No-Stress"),
])
def test_demo_recordings_have_varied_feature_based_predictions(filename, expected):
    from explainability.shap_engine import SHAPEngine
    from pipeline.manager import ECGStressPipelineManager
    predictor = RandomForestPredictor()
    pipeline = ECGStressPipelineManager(
        CSVECGIngestor(250), NeuroKitPreprocessor(), NeuroKitHRVExtractor(),
        predictor, SHAPEngine(predictor.get_model()),
    )
    report = pipeline.run_pipeline(str(ROOT / 'demo samples' / filename))
    assert report.prediction.is_demo
    assert report.prediction.prediction_label == expected
    assert 0 < report.prediction.stress_probability < 1
    assert len(report.prediction.top_shap_features) == 5
