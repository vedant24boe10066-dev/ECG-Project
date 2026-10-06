import os
import pandas as pd
import numpy as np
from ingestion.csv_ingestor import CSVECGIngestor
from preprocessing.filters import NeuroKitPreprocessor
from features.hrv_extractor import NeuroKitHRVExtractor
from models.classifiers import RandomForestPredictor
from explainability.shap_engine import SHAPEngine
from pipeline.manager import ECGStressPipelineManager
from reporting.json_reporter import JSONReportGenerator

def main():
    print("🚀 Initializing ECG Stress Detection Pipeline...")
    
    # 1. Initialize the components
    ingestor = CSVECGIngestor(sampling_rate=250.0)
    preprocessor = NeuroKitPreprocessor()
    hrv_extractor = NeuroKitHRVExtractor()
    
    # The built-in model uses explicitly synthetic HRV teaching scenarios.
    predictor = RandomForestPredictor(model_path=os.environ.get("ECG_MODEL_PATH"))
    if predictor.is_demo:
        print("⚙️ Demo mode: synthetic teaching scenarios; not validated stress findings.")
    shap_engine = SHAPEngine(model=predictor.get_model())

    # 4. Assemble the Pipeline Manager
    pipeline = ECGStressPipelineManager(
        ingestor=ingestor,
        preprocessor=preprocessor,
        hrv_extractor=hrv_extractor,
        predictor=predictor,
        shap_engine=shap_engine
    )
    
    # 5. Run the Pipeline on an ECG file
    target_file = "sample_ecg.csv"
    
    # Create a dummy CSV if it doesn't exist just to test
    if not os.path.exists(target_file):
        print(f"⚠️ {target_file} not found. Creating a synthetic 30-second ECG CSV...")
        from ingestion.mock_generator import MockECGGenerator
        mock_signal = MockECGGenerator.generate(duration_seconds=30, sampling_rate=250)
        pd.DataFrame({"ECG": mock_signal.signal_data}).to_csv(target_file, index=False)
        print(f"✅ Created {target_file}")
    
    print(f"📊 Running pipeline on: {target_file}")
    report = pipeline.run_pipeline(target_file)
    
    # 6. Save Report
    report_path = JSONReportGenerator.save_report(report)
    print(f"\n✅ Pipeline Complete! Report saved to: {report_path}")
    print("\n--- Report Preview ---")
    print(f"ID: {report.report_id}")
    print(f"Prediction: {report.prediction.prediction_label} (Confidence: {report.prediction.confidence_score:.2f})")
    print(f"Top Contributing Feature: {report.prediction.top_shap_features[0].feature_name} (Impact: {report.prediction.top_shap_features[0].impact_direction})")

if __name__ == "__main__":
    main()
