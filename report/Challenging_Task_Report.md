# ECG Based Stress Detection Using Heart Rate Variability and Explainable Machine Learning

**Challenging Task Report**  
**Report date:** 6 October 2026

## Project Members

| Student name | Registration number |
|---|---|
| To be provided for every project member | To be provided for every project member |

**Member details pending:** Replace this placeholder with one row for each member when the names and registration numbers are supplied. Include every member before final submission.

## Abstract

This project implements a local application for analysing electrocardiogram (ECG) recordings through heart rate variability (HRV) features and explainable machine learning. It integrates CSV, EDF, and WFDB ingestion, signal cleaning, R-peak detection, ten HRV measurements, binary classification, SHAP explanations, interactive graphs, and downloadable JSON and PDF reports. Random Forest, support vector machine, Logistic Regression, and Extra Trees classifiers can analyse the same extracted features; XGBoost is implemented as an optional alternative. The central engineering challenge is to maintain consistent signal processing, model inputs, explanations, and reporting while handling invalid recordings, unavailable measurements, and unavailable model runtimes. Recorded evaluation on ten synthetic ECG signals at 250 Hz shows successful end-to-end processing and report generation. The Random Forest demonstration produced four Stress and six No-Stress outputs, with recorded single-model analysis times between 0.038 and 0.189 seconds. Four available classifiers agreed on the binary outputs for these examples but assigned different probabilities. These findings establish the functionality of the prototype, rather than clinical stress-detection accuracy: its default classifiers use synthetic HRV training scenarios and illustrative labels. Genuine stress annotations, participant-independent evaluation, and validation of short-recording HRV measurements remain necessary for research conclusions about real stress detection.

**Keywords:** ECG, heart rate variability, stress classification, machine learning, SHAP, NeuroKit2, explainability.

## 1 Introduction

### 1.1 Background

ECG records the electrical activity of the heart. The time between successive R-peaks provides a sequence of RR intervals from which HRV measurements can be calculated. These measurements describe variation in cardiac timing rather than simply the average number of beats per minute. Standardised HRV analysis includes time-domain and frequency-domain measurements, with recording conditions and duration affecting interpretation [1].

Physiological stress recognition is a possible application of cardiac measurements. However, a physiological change is not a unique stress label: an application's classification must be evaluated against independently obtained annotations. The WESAD research dataset illustrates this distinction by combining physiological recordings with experimental affective conditions and participant self-reports [3]. This project explores the software pipeline needed to move from a raw ECG recording to a transparent classification result.

### 1.2 Problem Statement

A useful prototype must do more than return a Stress or No-Stress label. It must read different recording formats, preserve sampling-rate information, reject unsuitable input, detect beats, calculate a consistent feature vector, and explain how a model used those features. It must also distinguish a measurement from an unavailable value and a demonstration prediction from a validated finding. The challenging task addressed here is the integration of these requirements into a reproducible application with model comparison and readable reports.

### 1.3 Objectives

The project aims to:

1. Load local ECG recordings from CSV, EDF, and WFDB sources.
2. Clean recordings, detect R-peaks, and derive RR intervals.
3. Extract ten HRV features spanning time, frequency, and nonlinear domains.
4. Apply selectable classifiers to the same feature representation.
5. Explain individual model outputs using SHAP feature attributions.
6. Present recording analytics and export structured JSON and printable PDF results.
7. Preserve explicit errors, missing-value warnings, and demonstration status throughout the application.

### 1.4 Scope

The implemented system processes uploaded or locally stored recordings. It includes a command-line pipeline and a FastAPI browser interface. The waveform display is a replay of recorded data; it does not acquire a live sensor stream. Hardware integration, continuous wearable monitoring, and a completed clinical evaluation are outside the implemented scope.

Although the workspace folder is named EEG Project, the source code, interfaces, and data pipeline implement ECG analysis. This report therefore concerns cardiac signal processing, rather than electroencephalography or brain-signal analysis. The current classification task is binary.

## 2 Literature Review

### 2.1 Standards for HRV Measurement

The joint European Society of Cardiology and North American Society of Pacing and Electrophysiology Task Force established terminology and methods for HRV analysis [1]. Its guidance makes recording duration and comparability important considerations. This project follows the broad organisation into time-domain and spectral measurements, but its short demonstration recordings should not be assumed equivalent to standard five-minute short-term assessments. The practical implication is that a software-generated number still requires an appropriate recording protocol.

### 2.2 Reproducible Physiological Signal Processing

Makowski and colleagues introduced NeuroKit2 as an open-source toolbox for neurophysiological signal processing [2]. Its high-level functions provide a practical foundation for ECG cleaning, peak detection, and HRV calculation. The project uses these functions instead of implementing a new detector or spectral estimator. This reduces implementation effort, while making the installed library version and selected defaults part of the methodology. Reusing a published toolbox does not itself verify detector accuracy on the project's recordings.

### 2.3 Stress Datasets and Experimental Labels

Schmidt and colleagues introduced WESAD, a multimodal wearable dataset containing physiological measurements associated with neutral, stress, and amusement conditions, together with self-reports [3]. It demonstrates the role of an experimental protocol in establishing stress labels. In this project, generated waveforms and synthetic HRV rows support development and demonstration. They do not replace a participant dataset. WESAD is a relevant candidate for future validation; it was not used to produce the reported project results.

### 2.4 Machine Learning for Tabular Features

Breiman's Random Forest method combines randomised decision trees into an ensemble [4]. Its use here permits nonlinear decisions over a compact HRV feature vector. The application also includes an RBF support vector machine, Logistic Regression, and Extra Trees to compare alternative decision functions. This comparison is useful for inspecting model behaviour on common inputs, but selecting a superior stress classifier requires an independent labelled evaluation rather than agreement on demonstration examples.

### 2.5 Explainable Predictions

Lundberg and Lee proposed SHAP as an additive feature-attribution framework for explaining individual model predictions [5]. The project uses TreeSHAP for supported tree ensembles and KernelSHAP for the scaled SVM and Logistic Regression pipelines. Attributions indicate which features move a model output relative to its reference behaviour. They explain the classifier's calculation; they do not establish that a physiological feature caused stress. Correlated HRV features and the chosen background data also affect interpretation.

### 2.6 Limits of Spectral Interpretation

Billman examined the assumptions behind treating LF/HF as a direct measure of cardiac sympathetic–parasympathetic balance and argued that this interpretation is unreliable [6]. Consequently, this report treats LF/HF as a computed feature rather than a standalone stress marker. Its contribution to a prediction must be assessed alongside signal quality, recording conditions, other measurements, and the model's training evidence.

### 2.7 Position of the Present Work

The project's contribution is the integration of existing processing and modelling methods into a local, explainable workflow. It combines multi-format ingestion, explicit missing-value handling, selectable classifiers, reusable signal analytics, and report export. It does not claim a new HRV estimator or a newly validated stress biomarker. The literature identifies the main gap between the working prototype and a research-validated system: reliable labels and an evaluation design that measures generalisation to unseen participants.

## 3 Methodology

### 3.1 System Architecture

The code separates ingestion, preprocessing, feature extraction, classification, explanation, reporting, and orchestration into modules. The main processing sequence is:

**ECG file → input validation → cleaning → R-peak detection → RR intervals → HRV features → classification → SHAP explanation → graphs and reports**

`ECGStressPipelineManager` coordinates the command-line workflow. The browser API executes corresponding stages while streaming progress messages. In comparison mode, preprocessing and HRV extraction run once, and each available classifier receives the same features. An unsuccessful alternative is retained as an error entry so that successful model results remain visible.

### 3.2 Input Formats and Validation

CSV input requires an `ECG` column containing one numeric raw signal sample per row. The user supplies its sampling rate. EDF ingestion reads channel metadata and prefers a channel containing ECG or EKG in its label, otherwise falling back to the first channel. WFDB ingestion uses the paired header and data record and reads the first signal channel. Channel selection should therefore be checked for multi-channel recordings.

The shared signal schema requires finite one-dimensional values, a whole-number sampling rate of at least 50 Hz, and at least ten seconds of data. Constant recordings are rejected. Preprocessing additionally requires at least ten detected R-peaks. These conditions are operational safeguards, not proof that all HRV features are statistically reliable.

### 3.3 Signal Cleaning and R Peak Detection

`NeuroKitPreprocessor` calls `nk.ecg_clean` and `nk.ecg_peaks` with `method='neurokit'` and enables peak artifact correction. The installed cleaning implementation uses a fifth-order Butterworth high-pass stage at 0.5 Hz followed by powerline filtering, whose default frequency is 50 Hz. This agrees with the documented NeuroKit implementation [7].

The current code does not explicitly configure the 0.5–45 Hz bandpass or the Pan–Tompkins detector described in early design notes. The implemented method is therefore reported as the default NeuroKit pipeline. After detecting sample indices rᵢ, successive intervals are calculated as:

`RRᵢ in milliseconds = 1000 × (rᵢ₊₁ − rᵢ) / sampling rate`

At 250 Hz, sample indices are spaced 4 milliseconds apart. Accurate sampling-rate metadata is essential because an incorrect value scales every interval and changes the derived measurements.

### 3.4 HRV Feature Extraction

The extractor calls `nk.hrv` and maps its output to ten named features [8]. It validates that R-peak indices are finite integers, strictly increasing, and within the recording boundaries.

| Domain | Features | Meaning |
|---|---|---|
| Time | SDNN, RMSSD, pNN50, MeanRR | Interval dispersion, successive differences, percentage of differences exceeding 50 ms, and mean interval |
| Frequency | LF, HF, LF_HF_Ratio | Spectral measurements in low- and high-frequency bands and their ratio |
| Nonlinear | SD1, SD2, SampEn | Poincaré dispersion and sample entropy |

For N intervals, RMSSD is the square root of the mean squared successive interval differences. pNN50 is the percentage of successive absolute differences exceeding 50 milliseconds. SDNN, RMSSD, MeanRR, SD1, and SD2 use milliseconds; pNN50 uses percent; LF/HF and sample entropy are dimensionless. The default frequency bands are 0.04–0.15 Hz for LF and 0.15–0.40 Hz for HF. The installed spectral function defaults to Welch estimation and spectrum normalisation. Its returned LF and HF values should therefore not be assumed to be unnormalised absolute powers without recording the spectral configuration.

Unavailable or nonfinite measurements become `None`, which serialises as JSON `null`, and generate warnings. Demonstration inference substitutes zero for such features while preserving nulls in measured summaries. Non-demo inference rejects unavailable features. This policy keeps the demonstration operational but can influence its predictions and explanations.

### 3.5 Feature Selection Status

`HRVFeatureSelector` implements `SelectKBest` using mutual information or an ANOVA F score. However, the current API and main pipeline pass all ten features directly to the classifier. Feature selection is an available component, rather than an active stage in the reported demonstration. A future evaluation should fit any selector exclusively within training folds to prevent information leakage.

### 3.6 Demonstration Training and Classification

The default training generator creates 600 synthetic HRV rows using a random seed of 42. It samples RMSSD, SDNN, and MeanRR ranges, constructs related variability measurements, and generates additional spectral and entropy features. The illustrative score is:

`score = (25 − RMSSD) / 12 + (20 − SDNN) / 16 + (800 − MeanRR) / 500`

An illustrative probability is calculated as `0.08 + 0.84 / (1 + exp(−score))`. Binary teaching labels are sampled from this probability. These constants are demonstration settings, not physiological diagnostic thresholds.

| Classifier | Implemented configuration | Explanation |
|---|---|---|
| Random Forest | 100 trees; seed 42; minimum leaf size 8 in demo fitting | TreeSHAP |
| SVM | StandardScaler followed by RBF SVC with probability output | KernelSHAP |
| Logistic Regression | StandardScaler followed by Logistic Regression; maximum 2000 iterations | KernelSHAP |
| Extra Trees | 100 trees; seed 42; minimum leaf size 8 in demo fitting | TreeSHAP |
| XGBoost | Optional 100-estimator classifier; runtime dependent | TreeSHAP with a training background |

The models train lazily on first use and are reused afterwards. A probability greater than 0.5 produces Stress; other values produce No-Stress. The displayed confidence is `max(p, 1 − p)`. It is the model's assigned class probability, rather than measured accuracy or established clinical confidence. Saved artifacts retain the estimator, feature order, demo status, and background data. Incompatible feature schemas cause explicit errors rather than automatic retraining.

### 3.7 Explainability and Visual Analytics

SHAP explanations retain the five features with the largest absolute attribution values. Positive attributions increase the explained Stress output; negative values decrease it. KernelSHAP uses up to 24 seeded training rows as background and a coalition budget of 2^min(number of features, 10). A five-feature display is a ranking, not a complete additive reconstruction of all ten features.

Recording graphs show the waveform, interval-derived heart rate, RR intervals, and consecutive-RR Poincaré pairs. Instantaneous heart rate is `60000 / RRᵢ`. The displayed mean heart rate is calculated as `60000 / mean(RR)`, while extrema use individual intervals. Chart payloads are limited to 1500 points, but summary statistics use all intervals. Scatter sampling preserves original adjacent interval pairs.

### 3.8 Reporting and Evaluation Procedure

Pydantic schemas organise recording metadata, model identity, prediction, key HRV summaries, explanations, timings, warnings, comparisons, and signal analytics. JSON supports inspection and reuse; ReportLab creates downloadable PDF reports. Uploads are processed in a temporary session directory and cleaned up after the API request. Processing runs locally, but local execution alone should not be equated with a completed privacy or security assessment.

Evaluation uses three complementary forms of evidence: the ten stored synthetic-signal validation records, the four-model comparison records, and the existing automated regression suite. The demo recordings are all sampled at 250 Hz: eight last 60 seconds, one lasts 90 seconds, and one lasts 120 seconds. They vary simulated heart rate, variability, noise, and baseline drift. Recorded timings are observations from those runs, not a controlled performance benchmark. Classification accuracy, F1, and ROC-AUC cannot be estimated from these examples without genuine reference labels.

## 4 Results

### 4.1 End to End Processing

All ten stored demonstration records are marked as passing API analysis [10]. Each includes HRV extraction, a demo prediction, and five SHAP attributions. R-peak counts range from 60 to 130. Recorded single-model analysis times range from 0.038 to 0.189 seconds, with a mean of 0.066 seconds and median of 0.0495 seconds. These timings exclude any claim about all-model comparison latency and should not be extrapolated to long recordings or other computers.

### 4.2 HRV Measurements and Random Forest Outputs

The following values come from `demo samples/validation_results.json`. SDNN and RMSSD are rounded to two decimal places; probabilities are percentages.

| Sample | Duration s | R peaks | SDNN ms | RMSSD ms | Demo output | Stress probability |
|---|---:|---:|---:|---:|---|---:|
| 01 Baseline 70 bpm | 60 | 70 | 44.69 | 48.01 | No-Stress | 15.59% |
| 02 Slower 60 bpm | 60 | 60 | 50.61 | 62.02 | No-Stress | 10.08% |
| 03 Faster 90 bpm | 60 | 89 | 5.31 | 5.18 | Stress | 83.97% |
| 04 Faster 105 bpm | 60 | 104 | 4.57 | 4.17 | Stress | 84.04% |
| 05 Low variability 75 bpm | 60 | 74 | 5.56 | 5.94 | Stress | 85.24% |
| 06 Higher variability 75 bpm | 60 | 75 | 54.91 | 55.70 | No-Stress | 7.32% |
| 07 Light noise 70 bpm | 60 | 70 | 40.97 | 42.48 | No-Stress | 18.43% |
| 08 Baseline wander 70 bpm | 60 | 70 | 36.67 | 40.17 | No-Stress | 19.14% |
| 09 Longer 80 bpm | 90 | 119 | 11.53 | 11.16 | Stress | 88.54% |
| 10 Longer variable 65 bpm | 120 | 130 | 55.75 | 62.98 | No-Stress | 7.09% |

The two 75 bpm scenarios are particularly useful for demonstration. Their nominal simulator rates are equal, while their measured RMSSD values differ substantially. The lower-variability example receives a higher illustrative Stress probability. This is consistent with the synthetic teaching rule; it is not evidence that the rule identifies stress in people.

Sample 08 has an unavailable SampEn value and an explicit warning. Its successful analysis demonstrates missing-measurement handling, rather than proving that entropy was reliably estimated. Its demo classifier uses the configured placeholder.

### 4.3 Comparison Across Classifiers

The stored comparison contains ten inputs analysed by four available models. All four agree on each binary label, producing four Stress and six No-Stress outputs each. Probability estimates differ:

| Model | Sample 03 Stress probability | Sample 09 Stress probability | Stress outputs among ten |
|---|---:|---:|---:|
| Random Forest | 83.97% | 88.54% | 4 |
| SVM RBF | 70.16% | 62.39% | 4 |
| Logistic Regression | 83.33% | 75.97% | 4 |
| Extra Trees | 82.61% | 78.85% | 4 |

Agreement on these inputs does not establish accuracy. All four models use the same synthetic teaching distribution. A higher probability also does not identify a better classifier. XGBoost has no result in this comparison; project documentation identifies its local OpenMP runtime dependency as unavailable.

### 4.4 Report Generation and Software Verification

The stored PDF validation summary lists ten successful report exports, each with three pages and four compared models. Matching JSON artifacts are present. These are per-recording application reports, separate from this academic report.

The regression suite checks input rejection, missing measurements, saved-model schemas, model selection, comparison failures, PDF content, and interval analytics [11]. Verification on 6 October 2026 returned **59 passed tests in 87.39 seconds**, with 44 library and application warnings, including short-recording HRV warnings and deprecation notices. The Node interface checks also passed for model comparison, downloads, demo notices, null measurements, stream handling, waveform rendering, and Clear behaviour. Automated software tests establish expected program behaviour; they do not measure clinical validity or R-peak detection sensitivity against annotated beats.

### 4.5 Research Performance Metrics

The benchmark module contains cross-validation code for accuracy, F1, and ROC-AUC for Random Forest, SVM, and XGBoost. No completed participant-labelled benchmark results are present in the evaluated artifacts. Consequently, no stress-detection accuracy, sensitivity, specificity, F1, ROC-AUC, or confusion matrix is reported here. Likewise, the observed R-peak counts are not detector precision or recall measurements because no reference beat annotations accompany the demo evaluation.

## 5 Discussion

### 5.1 Engineering Outcomes

The system demonstrates a coherent path from raw ECG to an interpretable application report. Its main strengths are modular processing, a consistent feature schema, explicit demo status, and reuse of extracted features across models. The comparison interface lets users inspect the effect of classifier choice while holding the signal analysis constant. Structured errors and warnings make failures more understandable than silently returning a plausible label.

The handling of missing measurements is especially relevant to physiological data. Preserving nulls avoids confusing an unavailable result with a measured zero. However, demo-only zero substitution remains an approximation. The resulting attribution may partly explain the placeholder, so an explanation attached to a missing feature should not be interpreted as a physiological observation.

### 5.2 Challenging Aspects of Implementation

The first challenge is aligning heterogeneous file formats into a common signal structure. CSV requires explicit sample-rate input, whereas EDF and WFDB supply metadata and use different channel conventions. The second is maintaining beat indices, interval units, and feature names consistently across processing and inference. A mistake in any of these transformations can create believable but incorrect results.

The third challenge is explaining several model families through a common interface. Tree and kernel explainers require different setup, and non-tree models require a retained training background. The fourth is coordinating browser progress, partial comparison failures, and export schemas. Report metadata must continue to identify the selected model and any unavailable measurements after a user switches views or downloads a file.

These challenges make the project suitable as an integration-focused challenging task. They also explain why a working interface and successful tests must be assessed separately from scientific performance.

### 5.3 Scientific Limitations

Synthetic training is the largest limitation. Models learn an intentionally constructed relationship between HRV features and sampled labels, rather than an observed relationship between recordings and independently assessed stress. Some generated features are mathematically related, further constraining the synthetic feature distribution. Agreement across models therefore cannot establish generalisation to participants.

Recording length is another limitation. Most examples are one minute long, shorter than conventional five-minute short-term HRV recordings discussed in the standards [1]. Spectral and entropy outputs can depend on the number of beats and signal properties. The ten-second acceptance threshold is a programming guard and should not be presented as a universal measurement-duration recommendation.

The project also lacks a reported assessment of beat detection against reference annotations. Artifact correction is enabled, but no dedicated ectopic-beat assessment or clinical rhythm screening is demonstrated. LF/HF should receive cautious interpretation [6], and SHAP explanations remain explanations of the model rather than causal evidence [5].

### 5.4 Operational Limitations

The dependency list omits some directly imported application and ingestion packages, including FastAPI, Uvicorn, pyEDFlib, and WFDB. A clean installation is therefore not fully reproducible from the listed requirements alone. XGBoost's local runtime problem limits the available comparison. CPU-intensive extraction and explanation run within the asynchronous request flow, creating a potential responsiveness limitation for long recordings or concurrent requests.

First-use prediction timings can include synthetic model fitting, and SHAP timings include explainer setup. Formal runtime comparisons should warm up models, repeat runs, and report distributions with hardware and package versions. The default server binds to all network interfaces, so local computation should be distinguished from network access restrictions.

### 5.5 Future Work and Validation Plan

The next research stage should use a stress-labelled participant dataset with verified ECG channels, sampling rates, and experimental conditions. A resource such as WESAD provides a relevant starting point [3]. Recordings should be segmented using a documented window protocol and screened for signal quality and unsuitable rhythms.

Evaluation should separate participants between training and testing, preventing windows from one participant appearing on both sides. Scaling, feature selection, and any fitted missing-value treatment must remain inside the training folds. Hyperparameters should be selected using training-only validation, followed by one evaluation on a held-out test set. Report accuracy, precision, recall, F1, ROC-AUC, class counts, confusion matrices, and uncertainty estimates. Probability calibration should also be assessed before interpreting confidence scores.

Signal-processing validation should compare detected R-peaks with reference annotations using a stated matching tolerance. Robustness experiments should deliberately vary noise and artifacts, rather than infer robustness from successful uploads. Engineering improvements should complete dependency declarations, resolve optional runtime support, move heavy work to a suitable worker path, and measure repeated warm and cold execution times separately.

### 5.6 Overall Assessment

The project meets the practical objective of demonstrating local ECG analysis with multiple classifiers, explanations, visual analytics, and reports. Its recorded outcomes support a functional software prototype and an educational comparison of model behaviour. The remaining research objective is to determine whether its predictions generalise to real stress-labelled recordings under a controlled, participant-independent evaluation.

## 6 References

[1] Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: Standards of measurement, physiological interpretation and clinical use. *Circulation, 93*(5), 1043–1065. https://doi.org/10.1161/01.CIR.93.5.1043. [Publication record](https://pubmed.ncbi.nlm.nih.gov/8598068/).

[2] Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., Schölzel, C., & Chen, S. H. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods, 53*, 1689–1696. https://doi.org/10.3758/s13428-020-01516-y. [Author-hosted paper](https://dominiquemakowski.github.io/publication/makowski2021neurokit/makowski2021neurokit.pdf).

[3] Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. *Proceedings of the International Conference on Multimodal Interaction*. [Dataset and citation](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html).

[4] Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. [Publisher article](https://doi.org/10.1023/A:1010933404324).

[5] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. [Paper](https://arxiv.org/abs/1705.07874).

[6] Billman, G. E. (2013). The LF/HF ratio does not accurately measure cardiac sympatho-vagal balance. *Frontiers in Physiology, 4*, Article 26. [Article](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2013.00026/full).

[7] NeuroKit2 contributors. ECG cleaning implementation and documentation. Accessed 6 October 2026. [Official documentation](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ecg/ecg_clean.html). The installed project implementation was also inspected.

[8] NeuroKit2 contributors. HRV functions and frequency-analysis documentation. Accessed 6 October 2026. [Official documentation](https://neuropsychology.github.io/NeuroKit/functions/hrv.html). The installed project implementation was also inspected.

[9] ECG Stress Detection Prototype. Local project source and documentation, inspected 6 October 2026: `README.md`, `api.py`, `pipeline/manager.py`, `ingestion/`, `preprocessing/filters.py`, `features/`, `models/`, `explainability/shap_engine.py`, and `reporting/`.

[10] ECG Stress Detection Prototype. Synthetic recording validation and model comparison artifacts: `demo samples/README.md`, `demo samples/validation_results.json`, and `demo samples/model_comparison.json`.

[11] ECG Stress Detection Prototype. Export artifacts and regression checks: `output/pdf/validation_results.json`, matching recording reports, `tests/test_comparison_reports.py`, `tests/test_required_fixes.py`, `tests/test_model_selection.py`, `tests/test_signal_analytics.py`, `tests/test_pipeline.py`, and `tests/test_ui.cjs`.
