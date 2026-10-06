# UI Context & Report Visual Specifications

## Theme & Visual Language

The visual interface and report output follow a **Dark Technical Health-Tech Workspace** design system. High-contrast typography, crisp signal wave charts, physiological metric gauges, model benchmark comparisons, and distinct SHAP contribution bars provide intuitive visual diagnostics.

## Color Tokens

| Role                    | CSS Variable       | Hex Value   | Clinical / UI Functionality                      |
| ----------------------- | ------------------ | ----------- | ------------------------------------------------ |
| Dark Background Base    | `--bg-base`        | `#0B0F17`   | Main application canvas background               |
| Surface Panel           | `--bg-surface`     | `#161C26`   | Card container and chart surfaces                |
| Primary Text            | `--text-primary`   | `#F1F5F9`   | Headings, primary values, signal readings        |
| Muted Text              | `--text-muted`     | `#94A3B8`   | Subtitles, timestamps, unit labels (`ms`, `Hz`)  |
| Normal / Low Stress     | `--state-success`  | `#10B981`   | "No-Stress" classification / normal HRV green    |
| Elevated Stress         | `--state-warning`  | `#F59E0B`   | Moderate stress warning indicator                |
| High Stress             | `--state-error`    | `#EF4444`   | "High Stress" state / critical alert red         |
| Primary Accent          | `--accent-primary` | `#6366F1`   | Active model selection, tabs, action buttons     |
| SHAP Stress Suppressor  | `--shap-decrease`  | `#06B6D4`   | Cyan bar: Feature decreases predicted stress     |
| SHAP Stress Enhancer    | `--shap-increase`  | `#F43F5E`   | Rose bar: Feature increases predicted stress     |

## Report Layout & Dashboard Blueprint

```
┌────────────────────────────────────────────────────────────────────────┐
│  ECG Stress Detection Report — ID: REP-20260906-001                    │
├────────────────────────────────────────────────────────────────────────┤
│  [ Header Status Banner ]                                              │
│  Predicted State: STRESS DETECTED (Confidence: 89.4%)                  │
│  Top ML Model: XGBoost (Accuracy: 94.2%, F1-Score: 0.93)               │
├──────────────────────────────────┬─────────────────────────────────────┤
│  [ Panel 1: ECG & R-Peak Wave ]  │  [ Panel 2: HRV Feature Metrics ]   │
│  - Filtered Signal Trace (mV)    │  - SDNN: 34.2 ms (Low)              │
│  - Red Dots @ R-Peak Timestamps  │  - RMSSD: 21.8 ms (Low)             │
│  - R-R Interval distribution     │  - LF/HF Ratio: 2.85 (Sympathetic)  │
├──────────────────────────────────┴─────────────────────────────────────┤
│  [ Panel 3: Model Benchmark Comparison Matrix ]                        │
│  Model        | Accuracy | F1-Score | ROC-AUC | Inference Time       │
│  --------------------------------------------------------------        │
│  XGBoost      | 94.2%    | 0.93     | 0.96    | 12 ms                │
│  RandomForest | 91.8%    | 0.90     | 0.93    | 18 ms                │
│  SVM          | 88.5%    | 0.87     | 0.90    | 8 ms                 │
├────────────────────────────────────────────────────────────────────────┤
│  [ Panel 4: SHAP Physiological Feature Importance ]                    │
│  LF/HF Ratio (2.85)     ███████████████████████ (+0.32 Stress)         │
│  RMSSD (21.8 ms)        █████████████████       (+0.24 Stress)         │
│  SDNN (34.2 ms)         ████████████            (+0.18 Stress)         │
│  Sample Entropy (1.12)  ██████                  (-0.09 Stress)         │
└────────────────────────────────────────────────────────────────────────┘
```

## Charting & Data Visualizations

1. **Filtered ECG Waveform Chart**:
   - Time axis (seconds) vs Amplitude axis (mV).
   - Overlay markers showing R-peak detections.
2. **HRV Frequency Spectrum (Welch PSD)**:
   - Shaded band for VLF (0.003–0.04 Hz), LF (0.04–0.15 Hz), and HF (0.15–0.40 Hz).
3. **Model Benchmark Comparison Matrix**:
   - Tabular view comparing Accuracy, F1-Score, ROC-AUC, and latency across **Random Forest**, **SVM**, and **XGBoost**.
4. **SHAP Feature Contribution Summary Bar Plot**:
   - Horizontal bar chart sorting features by absolute SHAP impact.
   - Distinct color coding for feature values pushing predictions toward High Stress vs Low Stress.

## Dark Theme and Research Model Selection (2026-10-06)

The frontend defaults to dark mode with restrained green accents and a matching waveform grid. The upload form offers Random Forest, scaled RBF SVM, scaled Logistic Regression, and Extra Trees. A model availability endpoint also exposes optional XGBoost; unavailable runtimes are disabled rather than disrupting other classifiers.

Each API request captures its selected predictor and uses its own SHAP engine. All demo models share seeded synthetic HRV training data and preserve demo labeling. Results retain the model ID/name, explanation method, and timing. TreeSHAP explains RF/Extra Trees; KernelSHAP explains scaled SVM/logistic probabilities with saved training background data. Saved artifacts retain that background and feature ordering. The existing model-loading path remains available and additional model-specific environment paths are supported.

Verified 45 Python tests and JavaScript checks for dark styling, model-selection requests, replay, rendering, progress, errors, and Clear. Existing ten Random Forest demo scenario outcomes remain unchanged.

## Readability Pass (2026-10-06)

The dark interface now uses larger text, higher-contrast controls, and a wider results area. Prediction/confidence, model details, HRV measurements, and SHAP contributions precede the smaller waveform. HRV values use prominent two-column tiles with units; SHAP shows actual feature values, signed colored contributions, and emphasis on the largest contribution. Demo status remains visible in the prediction heading and explanatory notice. Analyze/Clear/file-selection controls have larger targets and stronger borders; input requirements are collapsible. Uploads, streamed progress, model switching, replay, nullable metrics, and cancellation remain unchanged. Existing JavaScript regression checks passed.

## Model Comparison and Report Downloads (2026-10-06)

- Compare all available models with one ingestion/preprocessing/HRV pass; preserve the selected main result. Per-model errors are represented alongside successful predictions.
- Store typed `ModelComparisonResult` entries in an optional `ECGStressReport.model_comparison` list, preserving compatibility with existing reports.
- Comparison rows show state, Stress probability, confidence, timing, and initialization flags. View actions switch displayed explanations locally, with no new analysis request.
- Add PDF and JSON download controls. `POST /api/report.pdf` validates the report schema and generates an attachment in memory, with shared HRV, per-model comparison/SHAP, and explicit demo notes.
- Tests: 56 Python checks passed, including comparison and downloadable PDF verification for all ten samples and per-alternative failure handling; JavaScript checks cover comparison requests, local model views, PDF/JSON download actions, and existing workflows.
- Ten three-page sample PDF reports and matching JSON results were generated under `output/pdf/`; representative overview and explanation pages were rendered and visually checked.
- Requested screenshot palette cannot yet be matched because the temporary screenshot path is blocked by macOS; awaiting a direct chat attachment.

## Requested Palette Applied (2026-10-06)

The user supplied the exact hex values, resolving the blocked screenshot reference. UI background is #080616, panels #1A1953, secondary controls/borders #162E93, and primary actions/progress #2F2FE4. Text uses contrasting off-white and pale blue; stress/error states retain a distinct coral indication. The enlarged typography, result hierarchy, comparison view, waveform, and PDF/JSON download functions are unchanged.

## Black Theme (2026-10-06)

User replaced the blue palette with a black theme. The UI now uses a #000000 background, #151515 panels, charcoal controls, white primary actions, and grayscale waveform/result highlights. Existing readability and analysis/comparison/download behavior remain in place.


### Recording graphs and state accents (2026-10-06)

Black theme now uses red for Stress and green for No-Stress, including results, comparison states, waveform and analysis graphs. Below the waveform, analysis displays average/range heart rate, mean RR interval, interval count, heart-rate and RR time-series charts, and a consecutive-RR (Poincare) scatter plot. Charts use detected beats for every supported input format; default classifiers remain clearly marked as demonstrations. Chart payloads are limited to 1,500 points; summary statistics use all intervals, and scatter pairs preserve original consecutive beats. Model switching recolors graphs without recalculating the signal. Clear and a new analysis hide previous graphs. Verified with all 10 demo samples, 59 Python tests, and Node UI checks.
