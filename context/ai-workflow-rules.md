# AI Workflow Rules

## Approach

This project is an end-to-end, offline-first health-tech application designed for local ECG stress detection. The architecture relies on a privacy-focused, zero-cloud execution strategy where data processing, feature extraction, ML inference, and explainability run entirely on the local machine without remote compute or external dependencies.

The backend is built as a modular, spec-driven Python pipeline where data ingestion, signal preprocessing, HRV feature extraction, ML inference, and SHAP explainability are strictly separated into isolated modules with clear interface boundaries.

---

### 1. Architectural Pipeline & Module Separation

The execution flow follows an isolated, step-by-step pipeline:

```
[Local ECG File] (CSV / EDF / WFDB)
       │
       ▼
 1. Ingestion Module  ───► Standardized Raw Signal Data Structure
       │
       ▼
 2. Preprocessing &   ───► Noise Filtering (Bandpass + Notch), R-Peak Detection,
    HRV Module             RR-Interval Extraction & HRV Metrics Computation
       │
       ▼
 3. ML & Inference    ───► Pre-trained XGBoost Model Local Evaluation & 
    Module                 TreeSHAP Feature Importance Analysis
       │
       ▼
 4. Report Builder    ───► Structured, Schema-Validated JSON Stress Report
```

1. **Ingestion Module (`ingestion/`)**:
   - Responsibilities: Load local ECG recording files (CSV, EDF, WFDB format), validate sampling rates, and output a standardized `RawECGSignal` data object.
   - Isolation: Must contain zero preprocessing or feature extraction logic.

2. **Signal Preprocessing & HRV Extraction Module (`preprocessing/`)**:
   - Responsibilities: Apply digital bandpass filtering (0.5 – 45 Hz) and powerline notch filtering (50/60 Hz). Perform R-peak detection (e.g., Pan-Tompkins algorithm) and derive R-R intervals. Calculate Time-Domain (SDNN, RMSSD, pNN50), Frequency-Domain (LF, HF, LF/HF ratio), and Non-Linear (Poincaré SD1/SD2, Sample Entropy) HRV metrics.
   - Isolation: Must be completely agnostic of ML models or file formats.

3. **ML Inference & SHAP Explainability Module (`inference/`)**:
   - Responsibilities: Load a pre-trained local XGBoost classifier (`.json` / `.joblib` model artifact). Format the extracted HRV metrics into model input vectors, perform stress probability prediction, and calculate TreeSHAP values for local model interpretability.
   - Isolation: Must operate solely on normalized HRV feature objects without raw signal processing dependencies.

4. **Report Builder & Serialization Module (`reporting/`)**:
   - Responsibilities: Aggregate metadata, stress prediction probabilities, confidence metrics, top-K contributing SHAP features, and key HRV metrics into a schema-validated, standardized JSON output.
   - Isolation: Responsible only for JSON formatting, schema validation, and storage.

---

### 2. Data Structures & Schemas

The pipeline uses strict Data Transfer Objects (Pydantic / Dataclass models) to enforce type safety between module boundaries:

- **`RawECGSignal`**:
  - `signal_data`: `np.ndarray` (1D array of voltage values in mV)
  - `sampling_rate`: `float` (in Hz)
  - `channel_name`: `str`
  - `file_path`: `str`
  - `duration_seconds`: `float`

- **`ProcessedECGSignal`**:
  - `filtered_signal`: `np.ndarray`
  - `r_peaks`: `np.ndarray` (sample indices of R-peaks)
  - `rr_intervals`: `np.ndarray` (in milliseconds)
  - `sampling_rate`: `float`

- **`HRVMetrics`**:
  - `time_domain`: `Dict[str, float]` (SDNN, RMSSD, pNN50, Mean RR)
  - `frequency_domain`: `Dict[str, float]` (LF, HF, LF_HF_Ratio)
  - `nonlinear`: `Dict[str, float]` (SD1, SD2, SampEn)
  - `feature_vector`: `Dict[str, float]` (Flattened key-value pair map for ML model input)

- **`SHAPAttribution`**:
  - `feature_name`: `str`
  - `shap_value`: `float`
  - `feature_value`: `float`
  - `impact_direction`: `str` ("increases_stress" | "decreases_stress")

- **`StressPredictionResult`**:
  - `prediction_label`: `str` ("Stress" | "No-Stress")
  - `stress_probability`: `float` (0.0 to 1.0)
  - `confidence_score`: `float`
  - `top_shap_features`: `List[SHAPAttribution]`

- **`ECGStressReport`**:
  - `report_id`: `str`
  - `timestamp`: `str` (ISO-8601 string)
  - `source_file`: `str`
  - `prediction`: `StressPredictionResult`
  - `key_hrv_summary`: `Dict[str, float]`
  - `pipeline_metadata`: `Dict[str, Any]`

---

### 3. Class Hierarchies & Component Blueprint

To ensure modularity and extensibility, the architecture is designed with explicit interfaces (Abstract Base Classes):

#### A. Ingestion Hierarchy
```python
class BaseECGIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> RawECGSignal: pass

class CSVECGIngestor(BaseECGIngestor): ...
class EDFECGIngestor(BaseECGIngestor): ...
class WFDBECGIngestor(BaseECGIngestor): ...
```

#### B. Preprocessing & HRV Hierarchy
```python
class BaseECGPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, raw_signal: RawECGSignal) -> ProcessedECGSignal: pass

class StandardECGPreprocessor(BaseECGPreprocessor): ...

class BaseHRVExtractor(ABC):
    @abstractmethod
    def extract_features(self, processed_signal: ProcessedECGSignal) -> HRVMetrics: pass

class BioSPPyHRVExtractor(BaseHRVExtractor): ...
```

#### C. Inference & Explainability Hierarchy
```python
class BaseStressPredictor(ABC):
    @abstractmethod
    def predict(self, features: HRVMetrics) -> StressPredictionResult: pass

class LocalXGBoostPredictor(BaseStressPredictor): ...

class BaseSHAPExplainer(ABC):
    @abstractmethod
    def explain(self, model: Any, features: HRVMetrics, top_k: int = 5) -> List[SHAPAttribution]: pass

class TreeSHAPExplainer(BaseSHAPExplainer): ...
```

#### D. Pipeline Orchestrator & Report Generator
```python
class JSONReportGenerator:
    def generate_report(
        self, 
        raw_signal: RawECGSignal, 
        prediction: StressPredictionResult, 
        hrv_metrics: HRVMetrics,
        output_path: str
    ) -> ECGStressReport: ...

class ECGStressPipelineManager:
    """Facade class tying Ingestion -> Preprocessing -> Inference -> Explainability -> JSON Output."""
    def __init__(
        self, 
        ingestor: BaseECGIngestor, 
        preprocessor: BaseECGPreprocessor, 
        hrv_extractor: BaseHRVExtractor,
        predictor: BaseStressPredictor,
        reporter: JSONReportGenerator
    ): ...
    
    def run_pipeline(self, file_path: str, output_json_path: str) -> ECGStressReport: ...
```

---

### 4. Implementation Guidelines & Incremental Verification

1. **Strict Offline Execution**: No HTTP/API requests, cloud storage, or external API calls are allowed anywhere in the pipeline code.
2. **Deterministic Inputs/Outputs**: Every module must be testable using local mock/sample files (e.g. synthetic ECG signals or sample `.csv` files).
3. **Spec-Driven Workflows**: Features must be built incrementally, verifying each interface unit (Ingestion unit tests -> Preprocessing unit tests -> ML & SHAP unit tests -> Integration test) against these context files.

## Scoping Rules

- Work on one feature unit at a time
- Prefer small, verifiable increments over large
  speculative changes
- Do not combine unrelated system boundaries in a
  single implementation step

## When to Split Work

Split an implementation step if it combines:

- [Concern one — e.g. UI changes and background task changes]
- [Concern two — e.g. Multiple unrelated API routes]
- [Concern three — e.g. Behavior not clearly defined in
  the context files]

If a change cannot be verified end to end quickly,
the scope is too broad — split it.

## Handling Missing Requirements

- Do not invent product behavior not defined in the
  context files
- If a requirement is ambiguous, resolve it in the
  relevant context file before implementing
- If a requirement is missing, add it as an open question
  in `progress-tracker.md` before continuing

## Protected Files

Do not modify the following unless explicitly instructed:

- [e.g. components/ui/* — generated UI library components]
- [e.g. Any third-party library internals]

## Keeping Docs in Sync

Update the relevant context file whenever implementation
changes:

- System architecture or boundaries
- Storage model decisions
- Code conventions or standards
- Feature scope

## Before Moving to the Next Unit

1. The current unit works end to end within its defined scope
2. No invariant defined in `architecture.md` was violated
3. `progress-tracker.md` reflects the completed work
4. `npm run build` passes
