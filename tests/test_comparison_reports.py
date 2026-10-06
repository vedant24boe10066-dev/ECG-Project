"""Compare each demo sample once and verify downloadable PDF content."""
import asyncio
import io
import json
from pathlib import Path

import pytest
from fastapi import UploadFile
from pypdf import PdfReader
from reporting.schemas import ECGStressReport

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / 'demo samples').glob('*.csv'))


@pytest.mark.parametrize('path', SAMPLES, ids=lambda path: path.name)
def test_comparison_and_pdf_for_every_demo_sample(path, tmp_path, monkeypatch):
    import api
    payload = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    calls = {'preprocess': 0, 'extract': 0}
    preprocess = api.preprocessor.preprocess
    extract = api.hrv_extractor.extract_features
    def counted_preprocess(raw):
        calls['preprocess'] += 1
        return preprocess(raw)
    def counted_extract(processed):
        calls['extract'] += 1
        return extract(processed)
    monkeypatch.setattr(api.preprocessor, 'preprocess', counted_preprocess)
    monkeypatch.setattr(api.hrv_extractor, 'extract_features', counted_extract)
    async def run():
        response = await api.analyze_ecg(files=[UploadFile(filename=path.name, file=io.BytesIO(payload))],
                                         sampling_rate=250, model_name='svm', compare_models=True)
        messages = [json.loads(line) async for line in response.body_iterator]
        assert messages[-1]['status'] == 'complete', messages[-1]
        report = ECGStressReport.model_validate(messages[-1]['data'])
        download = await api.download_pdf_report(report)
        return report, download
    report, download = asyncio.run(run())
    assert calls == {'preprocess': 1, 'extract': 1}
    charts = report.signal_analytics
    assert charts is not None
    assert charts.total_intervals == report.pipeline_metadata['r_peaks_detected'] - 1
    assert len(charts.time_seconds) == len(charts.rr_intervals_ms) == len(charts.heart_rate_bpm)
    assert len(charts.consecutive_rr_ms) == charts.total_intervals - 1
    assert all(abs(hr - 60000 / rr) < 1e-8 for hr, rr in zip(charts.heart_rate_bpm, charts.rr_intervals_ms))
    assert report.pipeline_metadata['model_id'] == 'svm'
    assert len(report.model_comparison) == len(api.MODEL_NAMES)
    active = [row for row in report.model_comparison if row.prediction]
    assert len(active) >= 4
    if 'xgboost' not in api.model_errors:
        assert any(row.model_id == 'xgboost' for row in active)
    for row in active:
        assert row.model_used
        assert row.prediction.is_demo
        assert 0 <= row.prediction.stress_probability <= 1
        assert len(row.prediction.top_shap_features) == 5
    selected = next(row for row in active if row.model_id == 'svm')
    assert report.prediction == selected.prediction
    assert download.media_type == 'application/pdf'
    assert download.body.startswith(b'%PDF-')
    assert 'attachment;' in download.headers['content-disposition']
    pdf = PdfReader(io.BytesIO(download.body))
    assert len(pdf.pages) >= 3
    content = '\n'.join(page.extract_text() for page in pdf.pages)
    assert path.name in content
    assert 'DEMONSTRATION' in content and 'not validated' in content
    assert 'Model comparison' in content and 'Feature explanations' in content
    for row in active:
        assert row.model_name in content
    assert not list((tmp_path / 'data/raw').iterdir())


def test_one_failed_alternative_does_not_hide_other_comparison_results(tmp_path, monkeypatch):
    import api
    class BrokenModel:
        def get_model(self):
            return None
        def predict(self, features):
            raise ValueError('Test model failure')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(api.alternative_predictors, 'svm', BrokenModel())
    async def run():
        response = await api.analyze_ecg(files=[UploadFile(filename='sample.csv', file=io.BytesIO(SAMPLES[0].read_bytes()))], compare_models=True)
        messages = [json.loads(line) async for line in response.body_iterator]
        return messages[-1]
    final = asyncio.run(run())
    assert final['status'] == 'complete'
    rows = final['data']['model_comparison']
    failed = next(row for row in rows if row['model_id'] == 'svm')
    assert failed['prediction'] is None and 'Test model failure' in failed['error']
    assert sum(row['prediction'] is not None for row in rows) >= 3
