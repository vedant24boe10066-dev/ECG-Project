import pytest
import pandas as pd
import numpy as np
import json
from ingestion.mock_generator import MockECGGenerator
from preprocessing.filters import NeuroKitPreprocessor
from features.hrv_extractor import NeuroKitHRVExtractor
from models.classifiers import RandomForestPredictor
from explainability.shap_engine import SHAPEngine
from pipeline.manager import ECGStressPipelineManager

def test_end_to_end_pipeline():
    # Setup dummy training data for RF Model and SHAP background
    np.random.seed(42)
    # create synthetic feature dataframe matching our HRV outputs
    dummy_features = ["SDNN", "RMSSD", "pNN50", "MeanRR", "LF", "HF", "LF_HF_Ratio", "SD1", "SD2", "SampEn"]
    X_train = pd.DataFrame(np.random.rand(100, 10), columns=dummy_features)
    y_train = np.random.randint(0, 2, size=100)
    
    # 1. Initialize ML Predictor and train dummy
    predictor = RandomForestPredictor(random_state=42)
    predictor.fit(X_train, y_train, demo=True)
    
    # 2. Initialize SHAP Engine
    shap_engine = SHAPEngine(model=predictor.get_model())
    
    # 3. Assemble Pipeline
    pipeline = ECGStressPipelineManager(
        ingestor=MockECGGenerator(),  # We bypass normal ingestion to use MockECGGenerator direct generate
        preprocessor=NeuroKitPreprocessor(),
        hrv_extractor=NeuroKitHRVExtractor(),
        predictor=predictor,
        shap_engine=shap_engine
    )
    
    # Override ingest for testing (since MockECGGenerator returns signal from generate())
    pipeline.ingestor.ingest = lambda file_path: MockECGGenerator.generate(duration_seconds=30, sampling_rate=250)
    
    # 4. Run Pipeline
    report = pipeline.run_pipeline("mock_patient_001.edf")
    
    # 5. Assertions
    assert report.report_id.startswith("REP-")
    assert report.prediction.prediction_label in ["Stress", "No-Stress"]
    assert 0.0 <= report.prediction.stress_probability <= 1.0
    assert len(report.prediction.top_shap_features) <= 5
    
    # Print JSON output
    print(report.model_dump_json(indent=2))

if __name__ == "__main__":
    test_end_to_end_pipeline()
