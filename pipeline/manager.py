import datetime
import uuid
import pandas as pd
from typing import Dict, Any
from ingestion.base import BaseECGIngestor
from preprocessing.base import BaseECGPreprocessor
from features.base import BaseHRVExtractor
from models.base import BaseStressPredictor
from explainability.shap_engine import SHAPEngine
from reporting.schemas import ECGStressReport

class ECGStressPipelineManager:
    """Facade orchestrating the offline-first ECG pipeline."""
    
    def __init__(
        self,
        ingestor: BaseECGIngestor,
        preprocessor: BaseECGPreprocessor,
        hrv_extractor: BaseHRVExtractor,
        predictor: BaseStressPredictor,
        shap_engine: SHAPEngine
    ):
        self.ingestor = ingestor
        self.preprocessor = preprocessor
        self.hrv_extractor = hrv_extractor
        self.predictor = predictor
        self.shap_engine = shap_engine

    def run_pipeline(self, file_path: str) -> ECGStressReport:
        # 1. Ingestion
        raw_signal = self.ingestor.ingest(file_path)
        
        # 2. Preprocessing & R-Peak Detection
        processed_signal = self.preprocessor.preprocess(raw_signal)
        
        # 3. HRV Feature Extraction
        hrv_metrics = self.hrv_extractor.extract_features(processed_signal)
        
        # 4. ML Inference
        prediction_result = self.predictor.predict(hrv_metrics.feature_vector)
        
        # 5. Explainability
        self.shap_engine.set_model(self.predictor.get_model(), getattr(self.predictor, "background_data", None))
        explain_features = self.predictor.prepare_features(hrv_metrics.feature_vector) if hasattr(self.predictor, "prepare_features") else hrv_metrics.feature_vector
        top_shap_features = self.shap_engine.explain(explain_features, top_k=5)
        prediction_result.top_shap_features = top_shap_features
        
        # 6. Report Generation
        report = ECGStressReport(
            report_id=f"REP-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            source_file=file_path,
            prediction=prediction_result,
            key_hrv_summary={
                "SDNN": hrv_metrics.time_domain.get("SDNN"),
                "RMSSD": hrv_metrics.time_domain.get("RMSSD"),
                "LF_HF_Ratio": hrv_metrics.frequency_domain.get("LF_HF_Ratio"),
                "Sample_Entropy": hrv_metrics.nonlinear.get("SampEn"),
            },
            pipeline_metadata={
                "is_demo": prediction_result.is_demo,
                "warnings": hrv_metrics.warnings,
                "sampling_rate_hz": raw_signal.sampling_rate,
                "duration_sec": raw_signal.duration_seconds,
                "r_peaks_detected": len(processed_signal.r_peaks)
            }
        )
        return report
