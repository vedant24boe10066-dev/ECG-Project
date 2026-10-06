# ECG-Based Stress Detection using HRV + ML

## Overview

This project is an offline-first, health-tech application for physiological stress detection using Electrocardiogram (ECG) signals and Heart Rate Variability (HRV) analysis. The application processes raw ECG recordings, cleans the signals, detects R-peaks, and computes comprehensive time-domain and frequency-domain HRV metrics. To maximize diagnostic accuracy and explainability, the pipeline performs automated feature selection, benchmarks multiple Machine Learning classifiers (Random Forest, Support Vector Machines, and XGBoost), and applies SHAP (SHapley Additive exPlanations) to identify which physiological markers contribute most to stress classification.

## Detailed Methodology Flow

```
┌───────────┐      ┌───────────────┐      ┌─────────────────┐      ┌────────────────┐
│ Raw ECG   │ ───► │ Signal        │ ───► │ R-Peak          │ ───► │ HRV Feature    │
│ Ingestion │      │ Preprocessing │      │ Detection       │      │ Extraction     │
└───────────┘      └───────────────┘      └─────────────────┘      └────────────────┘
                                                                           │
┌───────────────┐      ┌───────────────┐      ┌─────────────────┐          │
│ Explainable   │ ◄─── │ ML            │ ◄─── │ Feature         │ ◄────────┘
│ Prediction    │      │ Classification│      │ Selection       │
│ (SHAP Analysis│      │ (RF / SVM /   │      │ (ANOVA/Mutual   │
│ & JSON Report)│      │  XGBoost)     │      │  Information)   │
└───────────────┘      └───────────────┘      └─────────────────┘
```

1. **Raw ECG Ingestion**: Load local ECG files (`.csv`, `.edf`, `.dat`/WFDB) with automatic sampling rate detection.
2. **Signal Preprocessing**: Apply digital bandpass filtering (0.5 – 45 Hz) and powerline notch filtering (50/60 Hz) to eliminate baseline wander and high-frequency noise.
3. **R-Peak Detection**: Identify QRS complexes and extract precise R-peak timestamps using signal derivative and adaptive thresholding algorithms (e.g., Pan-Tompkins).
4. **HRV Feature Extraction**: Calculate time-domain (SDNN, RMSSD, pNN50, Mean RR), frequency-domain (LF, HF, LF/HF ratio via Welch PSD), and non-linear (Poincaré SD1/SD2, Sample Entropy) metrics.
5. **Feature Selection**: Apply statistical selection (ANOVA F-test / Mutual Information) to isolate the most discriminative HRV features and eliminate redundancy.
6. **ML Classification Comparison**: Compare pre-trained and dynamically trained classifiers (**Random Forest**, **SVM**, and **XGBoost**) for binary ("Stress" vs. "No-Stress") or multi-class stress evaluation.
7. **Explainable Prediction (SHAP)**: Compute SHAP feature attributions (TreeSHAP for XGBoost/RF, KernelSHAP for SVM) to quantify physiological drivers and generate a detailed JSON report.

## Goals

1. **High-Precision HRV Extraction**: Accurately detect R-peaks and extract standardized HRV markers from local raw ECG files.
2. **Comparative Model Evaluation**: Evaluate and benchmark Random Forest, SVM, and XGBoost on HRV feature subsets.
3. **Transparent & Explainable ML**: Use SHAP values to provide patient-specific feature attribution, highlighting the top physiological stress indicators.
4. **Offline & Privacy-Preserving**: Ensure 100% local execution without external API calls or cloud dependencies.

## Core User Flow

1. User selects or provides a local ECG file (`.csv`, `.edf`, `.dat`).
2. System ingests data and runs automated filtering and R-peak detection.
3. System extracts HRV time/frequency features and performs feature selection.
4. System passes features to the ML classifier suite (benchmarking RF, SVM, XGBoost) and selects/runs the top model.
5. System computes SHAP values for the prediction to isolate key physiological contributors.
6. System exports a structured `ECGStressReport` JSON object along with optional visual plots.

## Features

### Signal Processing & HRV Engine
- Multi-format ingestion (CSV, EDF, WFDB).
- Bandpass (0.5-45 Hz) and Notch (50/60 Hz) filtering.
- Pan-Tompkins R-peak detector and R-R interval derivation.
- Time-domain (SDNN, RMSSD, pNN50) & Frequency-domain (LF, HF, LF/HF) metrics.

### Machine Learning & Feature Selection
- Feature selection via Mutual Information / ANOVA F-score.
- Multi-model evaluation comparing **Random Forest**, **SVM**, and **XGBoost**.
- Performance metrics breakdown (Accuracy, F1-Score, ROC-AUC, Confusion Matrix).

### Explainability & Reporting
- SHAP value extraction (TreeSHAP & KernelSHAP).
- Top-K physiological stress feature attribution ranking.
- Structured, schema-validated JSON report serialization.

## Scope

### In Scope
- Local file ingestion and signal filtering.
- R-peak detection and HRV computation.
- Feature selection and model benchmark comparison (RF vs. SVM vs. XGBoost).
- SHAP feature contribution analysis.
- Offline JSON report generation.

### Out of Scope
- Real-time cloud streaming / telemetry sync.
- Direct hardware sensor Bluetooth integration.
- Continuous multi-day real-time wearable monitoring.

## Success Criteria

1. Pipeline runs end-to-end on local ECG test signals in under 5 seconds.
2. R-peak detection achieves high precision on standard benchmark ECG datasets.
3. Comparative classification suite produces accuracy, F1-score, and ROC-AUC metrics for RF, SVM, and XGBoost.
4. SHAP engine outputs ranked physiological feature contributions for every stress prediction.
5. Structured JSON report passes schema validation completely offline.
