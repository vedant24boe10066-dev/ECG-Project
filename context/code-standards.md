# Code Standards & Conventions

## General Principles

- **Single Responsibility Principle**: Keep modules focused on a single concern (ingestion, signal processing, feature selection, model inference, explainability, reporting).
- **Type Safety**: Enforce explicit Python type annotations across all function signatures, dataclasses, and Pydantic models.
- **Fail Early at Boundaries**: Validate input signal sample rates, array shapes, and file integrity immediately upon entry.
- **Reproducibility**: Explicitly set random seeds (`random_state=42`) for all ML estimators, feature selectors, and train/test splits.

## Python Coding Standards

- **Python Version**: Minimum Python 3.10 requirement. Use modern typing syntax (e.g., `list[str]`, `dict[str, float]`, `float | None`).
- **Data Transfer Objects**: Use Pydantic `BaseModel` or `@dataclass` for schema objects (`RawECGSignal`, `ProcessedECGSignal`, `HRVMetrics`, `StressPredictionResult`, `ECGStressReport`).
- **Exception Handling**: Inspect and raise domain-specific exceptions (e.g., `InvalidECGFileError`, `RPeakDetectionError`, `ModelInferenceError`) rather than bare `Exception`.

## Signal Processing Standards

- **Signal Data Type**: NumPy arrays with `float64` precision representing voltage in millivolts (mV).
- **Sampling Rate Validation**: Standardize signals to a minimum sampling frequency (e.g., 250 Hz or 500 Hz). Resample explicitly if input sampling rate diverges.
- **Unit Normalization**:
  - ECG Amplitude: millivolts (mV)
  - R-R Intervals: milliseconds (ms)
  - Frequency Bands: Low Frequency (LF: 0.04–0.15 Hz), High Frequency (HF: 0.15–0.40 Hz)

## Machine Learning & Feature Selection Standards

- **Model Interfaces**: Implement all ML models under `BaseStressPredictor` ABC.
- **Model Comparison Protocols**: Standardize evaluation metrics across **Random Forest**, **SVM**, and **XGBoost**:
  - Accuracy
  - Precision, Recall, Macro F1-Score
  - ROC-AUC
  - Confusion Matrix
- **Feature Selection**: Apply statistical selection (ANOVA F-test / Mutual Information) strictly within cross-validation folds to prevent data leakage during model benchmarking.

## Explainability (SHAP) Standards

- **Explainer Selection**: Use `shap.TreeExplainer` for tree-based models (XGBoost, Random Forest) and `shap.KernelExplainer` for SVM models.
- **Attribution Format**: Convert raw SHAP values into structured `SHAPAttribution` objects containing feature name, SHAP value, actual feature value, and impact direction ("increases_stress" or "decreases_stress").
- **Top-K Ranking**: Always rank SHAP feature contributions by absolute value descending.

## Project Structure & File Layout

```
ingestion/
  ├── base.py              # BaseECGIngestor ABC
  ├── csv_ingestor.py      # CSVECGIngestor
  ├── edf_ingestor.py      # EDFECGIngestor
  └── wfdb_ingestor.py     # WFDBECGIngestor
preprocessing/
  ├── base.py              # BaseECGPreprocessor ABC
  ├── filters.py           # Bandpass & Notch digital filtering
  └── rpeak_detector.py    # Pan-Tompkins R-peak detector
features/
  ├── base.py              # BaseHRVExtractor ABC
  ├── hrv_time.py          # SDNN, RMSSD, pNN50, Mean RR
  ├── hrv_freq.py          # LF, HF, LF/HF ratio (Welch PSD)
  ├── hrv_nonlinear.py     # Poincaré SD1/SD2, Sample Entropy
  └── feature_selector.py  # ANOVA F-test / Mutual Info selection
models/
  ├── base.py              # BaseStressPredictor ABC
  ├── xgboost_model.py     # LocalXGBoostPredictor
  ├── random_forest.py     # RandomForestPredictor
  ├── svm_model.py         # SVMPredictor
  └── benchmark.py         # Comparative Model Evaluator
explainability/
  ├── base.py              # BaseSHAPExplainer ABC
  └── shap_engine.py       # TreeSHAP & KernelSHAP attribution engines
reporting/
  ├── json_reporter.py     # JSONReportGenerator
  └── schemas.py           # Pydantic schemas (ECGStressReport, etc.)
pipeline/
  └── manager.py           # ECGStressPipelineManager facade
```
