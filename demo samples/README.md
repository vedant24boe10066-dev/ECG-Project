# ECG Demo Samples

These 10 CSV files contain reproducible **synthetic ECG waveforms**, generated locally with NeuroKit2. They are demonstration inputs, not patient recordings or labeled stress examples. All 10 passed the actual API analysis flow, including HRV extraction, demo inference, JSON report creation, and five SHAP attributions.

## How to showcase

1. Open http://127.0.0.1:8000 while the project server is running.
2. Set **Sampling Rate (Hz) to 250** for every file in this folder.
3. Upload **one CSV at a time** and click **Analyze Signal**.
4. Start with `01_baseline_70bpm.csv`, then compare the slower and faster heart-rate samples.
5. Compare `05_low_variability_75bpm.csv` with `06_higher_variability_75bpm.csv` to discuss beat-to-beat variation.
6. Use the light-noise and baseline-wander samples to discuss preprocessing.
7. Use the longer recordings to show another duration and explain that some HRV metrics depend on recording length.

The waveform display replays the uploaded signal. It is not live sensor acquisition. Each CSV has an `ECG` header with one raw signal sample per row.

| File | Duration | Simulator heart rate (BPM) | Demonstration | Detected R-peaks | Demo output (stress probability) |
|---|---:|---:|---|---:|---|
| `01_baseline_70bpm.csv` | 60 s | 70 | Higher-variability baseline scenario | 70 | No-Stress (16%) |
| `02_slower_60bpm.csv` | 60 s | 60 | Slower, more variable scenario | 60 | No-Stress (10%) |
| `03_faster_90bpm.csv` | 60 s | 90 | Faster, low-variability scenario | 89 | Stress (84%) |
| `04_faster_105bpm.csv` | 60 s | 105 | High-rate, low-variability scenario | 104 | Stress (84%) |
| `05_low_variability_75bpm.csv` | 60 s | 75 | Lower simulated beat-to-beat variability | 74 | Stress (85%) |
| `06_higher_variability_75bpm.csv` | 60 s | 75 | Higher simulated beat-to-beat variability | 75 | No-Stress (7%) |
| `07_light_noise_70bpm.csv` | 60 s | 70 | Higher variability with light noise | 70 | No-Stress (18%) |
| `08_baseline_wander_70bpm.csv` | 60 s | 70 | Higher variability with baseline drift | 70 | No-Stress (19%) |
| `09_longer_80bpm.csv` | 90 s | 80 | Longer, lower-variability scenario | 119 | Stress (89%) |
| `10_longer_variable_65bpm.csv` | 120 s | 65 | Two-minute recording with variable beats | 130 | No-Stress (7%) |

## Interpreting the results

The displayed Stress/No-Stress state, confidence, and SHAP contributions come from the application's **demo model**, trained on synthetic HRV scenarios with an illustrative lower-variability/higher-rate rule. These teaching thresholds are not clinically validated. The filenames describe simulation settings and do not establish a stress state. A faster heart rate alone does not label an input as stress. Changing signal characteristics does not guarantee different classification labels.

Unavailable HRV metrics are shown as “Unavailable” with warnings; an unavailable metric does not mean that the upload failed. Analysis time varies by machine and recording. Generation settings, seeds, detected peaks, measured HRV summaries, and API verification timings are recorded in `validation_results.json`.

## Model research

Use the Classification model selector to analyze the same file with Random Forest, SVM (RBF), Logistic Regression, or Extra Trees. All default models use shared synthetic teaching data. `model_comparison.json` records the ten samples' outputs across those four models; this is a comparison of demo predictions, not a benchmark of accuracy. The original outcome table above describes Random Forest. Models may disagree, which is useful for discussing algorithm behavior.
