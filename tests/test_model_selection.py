"""Research-model selection, saved artifacts, and explanation consistency."""
import asyncio
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from explainability.shap_engine import SHAPEngine
from features.hrv_extractor import NeuroKitHRVExtractor
from ingestion.csv_ingestor import CSVECGIngestor
from models.classifiers import RandomForestPredictor, SVMPredictor, LogisticRegressionPredictor, ExtraTreesPredictor, XGBoostPredictor
from preprocessing.filters import NeuroKitPreprocessor

ROOT = Path(__file__).resolve().parents[1]
FACTORIES = {"random_forest": RandomForestPredictor, "svm": SVMPredictor,
             "logistic_regression": LogisticRegressionPredictor, "extra_trees": ExtraTreesPredictor}
try:
    XGBoostPredictor()
except Exception:
    pass  # Optional native runtime is not present on every machine.
else:
    FACTORIES['xgboost'] = XGBoostPredictor


@pytest.fixture(scope='module')
def sample_features():
    features = []
    for name in ['01_baseline_70bpm.csv', '03_faster_90bpm.csv']:
        raw = CSVECGIngestor(250).ingest(str(ROOT / 'demo samples' / name))
        features.append(NeuroKitHRVExtractor().extract_features(NeuroKitPreprocessor().preprocess(raw)).feature_vector)
    return features


@pytest.mark.parametrize('model_id', FACTORIES)
def test_each_model_varies_and_explains_probability(model_id, sample_features):
    predictor = FACTORIES[model_id]()
    probabilities = []
    for features in sample_features:
        result = predictor.predict(features)
        assert result.is_demo
        probabilities.append(result.stress_probability)
        engine = SHAPEngine(predictor.get_model(), predictor.background_data)
        values = engine.explain(predictor.prepare_features(features), top_k=10)
        assert len(values) == 10
        assert np.isfinite([value.shap_value for value in values]).all()
        base = engine.explainer.expected_value
        if engine.method == 'TreeSHAP':
            base = base[list(predictor.get_model().classes_).index(1)]
        assert float(base) + sum(value.shap_value for value in values) == pytest.approx(result.stress_probability, abs=1e-5)
    assert probabilities[0] < .5 < probabilities[1]


@pytest.mark.parametrize('model_id', ['svm', 'logistic_regression', 'extra_trees'])
def test_alternative_saved_model_retains_schema_and_background(model_id, sample_features, tmp_path):
    predictor = FACTORIES[model_id]()
    expected = predictor.predict(sample_features[0])
    path = tmp_path / 'model.joblib'
    predictor.save_model(str(path))
    loaded = FACTORIES[model_id](model_path=str(path))
    reversed_features = dict(reversed(list(sample_features[0].items())))
    assert loaded.predict(reversed_features).stress_probability == expected.stress_probability
    assert loaded.is_demo and loaded.background_data is not None
    assert len(SHAPEngine(loaded.get_model(), loaded.background_data).explain(loaded.prepare_features(reversed_features))) == 5


def test_model_catalog_and_invalid_selection_do_not_create_uploads(tmp_path, monkeypatch):
    import api
    monkeypatch.chdir(tmp_path)
    catalog = asyncio.run(api.list_models())['models']
    assert {model['id'] for model in catalog if model['available']} >= set(FACTORIES)
    with pytest.raises(HTTPException, match='Unknown model'):
        asyncio.run(api.analyze_ecg(files=[UploadFile(file=io.BytesIO(b''), filename='empty.csv')], model_name='unknown'))
    monkeypatch.setitem(api.model_errors, 'xgboost', 'Optional runtime unavailable')
    with pytest.raises(HTTPException, match='Optional runtime unavailable'):
        asyncio.run(api.analyze_ecg(files=[UploadFile(file=io.BytesIO(b''), filename='empty.csv')], model_name='xgboost'))
    assert not (tmp_path / 'data/raw').exists()


@pytest.mark.parametrize('model_id', FACTORIES)
def test_api_selection_matches_report(model_id, tmp_path, monkeypatch):
    import api
    payload = (ROOT / 'demo samples/03_faster_90bpm.csv').read_bytes()
    monkeypatch.chdir(tmp_path)
    async def consume():
        response = await api.analyze_ecg(files=[UploadFile(filename='scenario.csv', file=io.BytesIO(payload))], sampling_rate=250, model_name=model_id)
        messages = [json.loads(chunk) async for chunk in response.body_iterator]
        assert messages[-1]['status'] == 'complete', messages[-1]
        return messages[-1]['data']
    report = asyncio.run(consume())
    assert report['pipeline_metadata']['model_id'] == model_id
    assert report['pipeline_metadata']['model_name'] == api.MODEL_NAMES[model_id]
    assert report['prediction']['is_demo']
    assert len(report['prediction']['top_shap_features']) == 5
    assert report['pipeline_metadata']['explanation_time_ms'] >= 0
    assert not list((tmp_path / 'data/raw').iterdir())


@pytest.mark.parametrize('failure_phase', ['initialization', 'explanation'])
def test_xgboost_tree_incompatibility_falls_back_to_probability_shap(sample_features, monkeypatch, failure_phase):
    if 'xgboost' not in FACTORIES:
        pytest.skip('Optional XGBoost runtime unavailable')
    predictor = XGBoostPredictor()
    result = predictor.predict(sample_features[0])
    def incompatible_tree(*args, **kwargs):
        raise ValueError('Simulated incompatible XGBoost tree format')
    if failure_phase == 'initialization':
        monkeypatch.setattr('explainability.shap_engine.shap.TreeExplainer', incompatible_tree)
    else:
        class IncompatibleExplainer:
            shap_values = incompatible_tree
        monkeypatch.setattr('explainability.shap_engine.shap.TreeExplainer', lambda *args, **kwargs: IncompatibleExplainer())
    engine = SHAPEngine(predictor.get_model(), predictor.background_data)
    values = engine.explain(predictor.prepare_features(sample_features[0]), top_k=10)
    assert engine.method == 'KernelSHAP'
    assert float(engine.explainer.expected_value) + sum(v.shap_value for v in values) == pytest.approx(result.stress_probability, abs=1e-5)
