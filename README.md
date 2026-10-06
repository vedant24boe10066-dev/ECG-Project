# ECG Stress Detection Prototype

Local ECG ingestion, NeuroKit2 cleaning and R-peak detection, ten HRV metrics, Random Forest inference, SHAP explanations, and JSON reports.

## Run the presentation demo

From the project directory, using the existing environment:

```sh
venv/bin/python api.py
```

Open `http://localhost:8000` and upload `sample_ecg.csv` at **250 Hz**. This sample is 30 seconds long. The waveform is a replay of the uploaded signal.

```sh
venv/bin/python run.py
venv/bin/python -m pytest tests -q
```

The CLI saves a report to `data/processed/`.

## Classification status

By default, classification uses a reproducible synthetic HRV teaching model. The UI and reports explicitly mark it as **demo mode**. Its state, confidence, and SHAP explanations are illustrative and are not validated stress findings.

For a trained model, fit `RandomForestPredictor` on a labeled feature DataFrame (0 = No-Stress, 1 = Stress), then call `save_model(path)`. Fit with `demo=True` when the training data is synthetic. The saved artifact retains feature order and demo status. Point the API at the saved model:

```sh
ECG_MODEL_PATH=/absolute/path/model.joblib venv/bin/python api.py
```

Only load model files you created or trust. Real stress classification still requires genuine stress labels and independent evaluation; supplying a trained artifact alone does not establish accuracy.

## Accepted inputs and unavailable metrics

CSV files must have an `ECG` header and one numeric raw signal sample per row. Other columns can be present. HRV summary tables and matrices of separate recordings are rejected rather than interpreted as a waveform. EDF and matching WFDB `.hea`/`.dat` pairs remain supported.

Signals must contain at least 10 seconds of finite one-dimensional data, use a whole-number sampling rate of at least 50 Hz, and yield at least 10 detected R-peaks. These are input safeguards, not guarantees that every HRV metric can be measured. Constant signals are rejected.

Unavailable HRV measurements remain `null` in JSON and appear as “Unavailable” with warnings in the UI. Demo inference alone uses explicit zero placeholders to keep the illustration running. A trained model rejects unavailable features rather than silently classifying them. Feature mismatches never trigger retraining.

## Remaining work outside the critical/high-priority fixes

- Complete stress-labeled model training and validation.
- Repair XGBoost's local OpenMP dependency before benchmarking all three classifiers.
- Complete the runtime dependency list for fresh installations.
- Optimize long-recording HRV extraction and move CPU work off the async request loop.

## Varied classroom samples and interface (2026-10-06)

The ten CSVs in `demo samples/` now produce four Stress and six No-Stress outputs in demo mode. The built-in demonstration model uses synthetic features in HRV units and soft illustrative labels derived from a lower-variability/higher-rate teaching rule, with larger tree leaves to avoid extreme certainty. These are not clinical thresholds or real stress labels. Uploaded filenames do not control predictions. The same built-in demonstration model is used by the API and CLI; externally trained models retain their existing loading path.

The interface uses local CSS, system fonts, and a native canvas waveform: uploads, progress, replay, measurements, confidence, warnings, and SHAP contributions remain available without external frontend downloads. Clear cancels an active analysis in the browser.

## Dark mode and research model selection

The interface now defaults to dark mode. Select **Random Forest**, **SVM (RBF)**, **Logistic Regression**, or **Extra Trees**, then click **Analyze recording**. Each result identifies its model. Changing the selection does not modify the model behind an existing result; analyze again to apply it. XGBoost is available only when its local runtime loads and otherwise appears disabled with an availability note.

All default demo models train on the same seeded synthetic HRV scenarios. SVM and Logistic Regression use fitted StandardScaler pipelines. Tree models use TreeSHAP; the scaled SVM and logistic models use KernelSHAP on the Stress probability with a fixed training background. Saved model artifacts now retain that background along with feature names and demo status. These comparisons illustrate algorithm behavior; they do not establish real stress-detection accuracy.

API endpoints: `GET /api/models` lists availability; `POST /api/analyze?sampling_rate=250&model_name=svm` selects a model. IDs are `random_forest`, `svm`, `logistic_regression`, `extra_trees`, and `xgboost`. Invalid or unavailable selections return an explicit error rather than switching silently to another model.

Optional saved-model environment variables: `ECG_MODEL_PATH` (Random Forest), `ECG_SVM_MODEL_PATH`, `ECG_LOGISTIC_MODEL_PATH`, `ECG_EXTRA_TREES_MODEL_PATH`, and `ECG_XGBOOST_MODEL_PATH`. Use the corresponding predictor's `fit()` and `save_model()` methods; non-demo inference continues to require complete finite HRV features.

Reports include model identity, SHAP method, and prediction/explanation timings. The first demo prediction timing includes fitting the synthetic model; later requests reuse it. The SHAP timing includes explainer setup. For formal speed comparisons, warm up models and measure repeated runs.

## One-upload comparison and report downloads

Click **Compare all models** after selecting a recording. The backend cleans the signal and extracts HRV once, then evaluates every available classifier against those same features. The comparison shows model state, Stress probability, confidence, prediction/SHAP timings, and initialization markers. Unavailable or failed alternatives stay visible with a reason; they do not hide successful models. The selected model remains the main result.

Click **View** in a comparison row to inspect that model's prediction and SHAP contributions without another upload. **Download PDF** creates a printable report with recording details, HRV measurements, every comparison result, and each successful model's explanations. **Save JSON** exports the full structured report. Both downloads preserve demo labeling and unavailable-metric warnings. Report generation runs in memory and does not create a server-side report history.

The comparison API uses `compare_models=true`; `POST /api/report.pdf` accepts an `ECGStressReport` JSON body. Ten tested sample PDFs and matching JSON reports are available in `output/pdf/`, with validation results in that folder. PDF runtime dependency: `reportlab`; PDF regression checks use `pypdf`.


### Recording graphs and state accents (2026-10-06)

Black theme now uses red for Stress and green for No-Stress, including results, comparison states, waveform and analysis graphs. Below the waveform, analysis displays average/range heart rate, mean RR interval, interval count, heart-rate and RR time-series charts, and a consecutive-RR (Poincare) scatter plot. Charts use detected beats for every supported input format; default classifiers remain clearly marked as demonstrations. Chart payloads are limited to 1,500 points; summary statistics use all intervals, and scatter pairs preserve original consecutive beats. Model switching recolors graphs without recalculating the signal. Clear and a new analysis hide previous graphs. Verified with all 10 demo samples, 59 Python tests, and Node UI checks.

## Share a public demo with Render

The repository includes `render.yaml` for a single Python web service. Push `requirements.txt` and `render.yaml` to GitHub, then sign in to Render and create a Blueprint from this repository. Review the Free plan before deploying. The resulting HTTPS `onrender.com` URL serves both the UI and API, so visitors do not need your Mac running.

For manual setup, create a **Web Service**, select Python 3 and branch `main`, use `pip install -r requirements.txt` as the build command, and `uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1` as the start command. Copy environment values from `render.yaml`.

Free services sleep after 15 minutes without traffic and can take about a minute to wake. The scientific libraries and model comparison may exceed a small instance's memory; verify every model with the demo samples on the deployed service and choose more memory if deployment logs show a memory limit. Hosting configuration has been prepared and checked locally; a live cloud deployment has not yet been verified.

Uploads are used temporarily during analysis and removed afterwards. Reports are downloaded directly; this service does not provide persistent report history. Demo predictions remain illustrative rather than validated stress assessments.
