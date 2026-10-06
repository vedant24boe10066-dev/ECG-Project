import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Dict, Any, Optional

# --- Pydantic allows arbitrary types (like np.ndarray) if configured ---
class ArbitraryTypesModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RawECGSignal(ArbitraryTypesModel):
    signal_data: np.ndarray = Field(description="1D array of voltage values in mV")
    sampling_rate: float = Field(description="Sampling rate in Hz")
    channel_name: str = "ECG"
    file_path: Optional[str] = None
    duration_seconds: float

    @model_validator(mode="after")
    def validate_signal(self):
        if not np.isfinite(self.sampling_rate) or self.sampling_rate < 50 or not self.sampling_rate.is_integer():
            raise ValueError("Sampling rate must be a whole number of at least 50 Hz.")
        if self.signal_data.ndim != 1 or not np.isfinite(self.signal_data).all():
            raise ValueError("ECG must be a one-dimensional signal containing only finite numeric values.")
        if len(self.signal_data) / self.sampling_rate < 10:
            raise ValueError("Upload at least 10 seconds of raw ECG samples, not a table of HRV measurements.")
        return self

class ProcessedECGSignal(ArbitraryTypesModel):
    filtered_signal: np.ndarray = Field(description="Filtered ECG signal")
    r_peaks: np.ndarray = Field(description="Sample indices of R-peaks")
    rr_intervals: np.ndarray = Field(description="R-R intervals in milliseconds")
    sampling_rate: float

class HRVMetrics(BaseModel):
    time_domain: Dict[str, Optional[float]] = Field(description="SDNN, RMSSD, pNN50, Mean RR")
    frequency_domain: Dict[str, Optional[float]] = Field(description="LF, HF, LF_HF_Ratio")
    nonlinear: Dict[str, Optional[float]] = Field(description="SD1, SD2, SampEn")
    warnings: List[str] = Field(default_factory=list)
    feature_vector: Dict[str, Optional[float]] = Field(description="Flattened features for ML model")

class SHAPAttribution(BaseModel):
    feature_name: str
    shap_value: float
    feature_value: float
    impact_direction: str = Field(description="'increases_stress' or 'decreases_stress'")

class StressPredictionResult(BaseModel):
    is_demo: bool = False
    prediction_label: str = Field(description="'Stress' or 'No-Stress'")
    stress_probability: float = Field(ge=0.0, le=1.0)
    confidence_score: float
    top_shap_features: List[SHAPAttribution]

class ModelComparisonResult(BaseModel):
    model_used: Optional[str] = None
    model_id: str
    model_name: str
    prediction: Optional[StressPredictionResult] = None
    prediction_time_ms: float = 0.0
    explanation_time_ms: float = 0.0
    explanation_method: Optional[str] = None
    included_training: bool = False
    error: Optional[str] = None


class SignalAnalytics(BaseModel):
    time_seconds: List[float]
    rr_intervals_ms: List[float]
    heart_rate_bpm: List[float]
    consecutive_rr_ms: List[List[float]]
    total_intervals: int
    mean_rr_ms: float
    mean_heart_rate_bpm: float
    min_heart_rate_bpm: float
    max_heart_rate_bpm: float


class ECGStressReport(BaseModel):
    report_id: str
    timestamp: str
    source_file: Optional[str] = None
    prediction: StressPredictionResult
    model_comparison: List[ModelComparisonResult] = Field(default_factory=list)
    signal_analytics: Optional[SignalAnalytics] = None
    key_hrv_summary: Dict[str, Optional[float]]
    pipeline_metadata: Dict[str, Any]
