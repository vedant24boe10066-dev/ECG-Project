# Architecture Context

## Stack

| Layer                | Technology / Library                 | Role                                                          |
| -------------------- | ------------------------------------ | ------------------------------------------------------------- |
| Language             | Python 3.10+                         | Core backend runtime                                          |
| Signal Processing    | SciPy, BioSPPy / NeuroKit2, NumPy    | Filtering, R-peak detection, frequency Welch PSD analysis     |
| Feature Engineering  | scikit-learn (`feature_selection`)   | Variance thresholding, ANOVA F-test, Mutual Information       |
| Machine Learning     | XGBoost, scikit-learn (RF, SVM)      | Comparative multi-model classification (XGBoost, RF, SVM)     |
| Explainable AI       | SHAP (`shap.TreeExplainer`)          | Feature importance & physiological attribution extraction     |
| Data Validation      | Pydantic v2 / Dataclasses            | Schema enforcement, type safety, DTO validation               |
| Serialization        | JSON                                 | Structured offline diagnostic report export                   |

## System Boundaries & Directory Map

- **`ingestion/`**: Ingests raw ECG recordings (`.csv`, `.edf`, `.dat`) and normalizes data into a `RawECGSignal` structure.
- **`preprocessing/`**: Applies digital filters (Bandpass 0.5–45Hz, Notch 50/60Hz), detects R-peaks, and derives R-R intervals (`ProcessedECGSignal`).
- **`features/`**: Computes time-domain (SDNN, RMSSD, pNN50), frequency-domain (LF, HF, LF/HF), and non-linear HRV metrics (`HRVMetrics`) and performs feature selection.
- **`models/`**: Manages model training, evaluation, and comparative inference across **Random Forest**, **SVM**, and **XGBoost** (`BaseStressPredictor`).
- **`explainability/`**: Computes SHAP values to quantify physiological feature contributions to stress predictions (`BaseSHAPExplainer`).
- **`reporting/`**: Assembles metadata, prediction results, HRV summaries, and SHAP feature rankings into standardized JSON reports (`JSONReportGenerator`).
- **`pipeline/`**: Orchestrates the end-to-end execution flow via `ECGStressPipelineManager`.

## Data Pipeline Architecture

```
                               ┌────────────────────────────────┐
                               │     Local ECG File Ingestion   │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │  Signal Preprocessing Module   │
                               │  - Bandpass Filter (0.5-45 Hz) │
                               │  - Notch Filter (50/60 Hz)     │
                               │  - Pan-Tompkins R-Peak Detect  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   HRV Feature Extraction &     │
                               │      Feature Selection         │
                               │  - Time / Frequency / NonLinear│
                               │  - ANOVA / Mutual Information  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │    Comparative ML Suite        │
                               │  ┌──────────┬───────┬────────┐ │
                               │  │ XGBoost  │  RF   │  SVM   │ │
                               │  └──────────┴───────┴────────┘ │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │    SHAP Explainability Engine  │
                               │  - Top Physiological Drivers   │
                               │  - Feature Impact Direction    │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │  Structured JSON Stress Report │
                               └────────────────────────────────┘
```

## Storage Model

- **Local File Ingestion**: Ingests read-only raw ECG files from local file paths.
- **Model Storage**: Pre-trained model artifacts stored locally in `models/artifacts/` (`.json` or `.joblib`).
- **Report Storage**: Generated JSON reports written to `outputs/reports/`.

## Invariants

1. **Strict Local Execution**: The entire pipeline must function 100% offline without remote server or cloud API calls.
2. **Raw Data Immutability**: Input ECG signal files are strictly read-only and must never be altered or overwritten.
3. **Module Isolation**: No ML inference logic may exist inside the preprocessing module, and no signal filtering logic may exist inside the inference module.
4. **Deterministic Feature Ordering**: The feature vector order passed to ML models and SHAP explainers must strictly match the model training schema.
5. **Report Schema Compliance**: Output reports must always validate against the `ECGStressReport` Pydantic schema before writing to disk.

## Review Corrections (2026-10-06)

- Random Forest uses a reproducible, explicitly labeled demo unless fitted or loaded from an artifact. `ECG_MODEL_PATH` selects an API model artifact.
- Saved artifacts contain the estimator, training feature names, and demo status. Inference preserves training feature order; mismatches do not retrain the model.
- Unavailable HRV measurements are nullable and accompanied by warnings. Demo inference uses explicit placeholders only; trained inference requires complete finite features.
- Raw input validation requires finite one-dimensional samples, at least 10 seconds of data, an integer sampling rate of at least 50 Hz, and at least 10 detected beats for extraction.
- CSV input requires an explicitly named ECG column; binary waveform inputs use the existing EDF/WFDB readers. The browser previews CSV waveform replay only.

## Demo Model and Frontend Revision (2026-10-06)

The default model trains once on seeded synthetic HRV teaching scenarios in the actual feature units, with soft simulated labels derived from an illustrative variability/rate rule. It remains marked as a demo; the rule is not clinically validated. Real model artifacts are loaded as before, and input filenames are never used to assign predictions. The CLI now uses the same default model as the API.

The frontend uses self-contained CSS, system fonts, and native canvas rendering in place of Tailwind, external fonts, and Chart.js. It preserves CSV waveform replay, multi-file WFDB uploads, streamed processing status, HRV measurements, SHAP ranking, confidence, and nullable-metric warnings. Clear aborts browser requests and invalidates pending preview callbacks.

## Dark Theme and Research Model Selection (2026-10-06)

The frontend defaults to dark mode with restrained green accents and a matching waveform grid. The upload form offers Random Forest, scaled RBF SVM, scaled Logistic Regression, and Extra Trees. A model availability endpoint also exposes optional XGBoost; unavailable runtimes are disabled rather than disrupting other classifiers.

Each API request captures its selected predictor and uses its own SHAP engine. All demo models share seeded synthetic HRV training data and preserve demo labeling. Results retain the model ID/name, explanation method, and timing. TreeSHAP explains RF/Extra Trees; KernelSHAP explains scaled SVM/logistic probabilities with saved training background data. Saved artifacts retain that background and feature ordering. The existing model-loading path remains available and additional model-specific environment paths are supported.

Verified 45 Python tests and JavaScript checks for dark styling, model-selection requests, replay, rendering, progress, errors, and Clear. Existing ten Random Forest demo scenario outcomes remain unchanged.

## Model Comparison and Report Downloads (2026-10-06)

- Compare all available models with one ingestion/preprocessing/HRV pass; preserve the selected main result. Per-model errors are represented alongside successful predictions.
- Store typed `ModelComparisonResult` entries in an optional `ECGStressReport.model_comparison` list, preserving compatibility with existing reports.
- Comparison rows show state, Stress probability, confidence, timing, and initialization flags. View actions switch displayed explanations locally, with no new analysis request.
- Add PDF and JSON download controls. `POST /api/report.pdf` validates the report schema and generates an attachment in memory, with shared HRV, per-model comparison/SHAP, and explicit demo notes.
- Tests: 56 Python checks passed, including comparison and downloadable PDF verification for all ten samples and per-alternative failure handling; JavaScript checks cover comparison requests, local model views, PDF/JSON download actions, and existing workflows.
- Ten three-page sample PDF reports and matching JSON results were generated under `output/pdf/`; representative overview and explanation pages were rendered and visually checked.
- Requested screenshot palette cannot yet be matched because the temporary screenshot path is blocked by macOS; awaiting a direct chat attachment.


### XGBoost enabled after libomp installation (2026-10-06)

XGBoost 3.4.1 now loads locally. Installed XGBoost/SHAP combination rejects interventional TreeSHAP during explanation; the engine falls back to KernelSHAP of the same stress probability, with the actual method shown in results and reports. Other tree models keep TreeSHAP. Regression coverage includes both explainer initialization and evaluation failure, probability additivity, and successful XGBoost rows for all ten comparison/PDF samples. All ten samples also succeeded with XGBoost selected as the primary model. Validation: 63 Python tests and Node UI checks passed.
