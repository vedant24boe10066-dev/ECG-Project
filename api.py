import os
import shutil
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from ingestion.csv_ingestor import CSVECGIngestor
from preprocessing.filters import NeuroKitPreprocessor
from features.hrv_extractor import NeuroKitHRVExtractor
from models.classifiers import RandomForestPredictor, SVMPredictor, LogisticRegressionPredictor, ExtraTreesPredictor, XGBoostPredictor
from explainability.shap_engine import SHAPEngine
from pipeline.manager import ECGStressPipelineManager
from reporting.signal_analytics import build_signal_analytics
from reporting.json_reporter import JSONReportGenerator
from reporting.schemas import ECGStressReport

app = FastAPI(title="ECG Stress Detection API")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global pipeline instance (In memory for prototype)
print("🚀 Initializing ECG Stress Detection Pipeline...")
preprocessor = NeuroKitPreprocessor()
hrv_extractor = NeuroKitHRVExtractor()
predictor = RandomForestPredictor(model_path=os.environ.get("ECG_MODEL_PATH"))

shap_engine = SHAPEngine(model=predictor.get_model())

MODEL_NAMES = {
    "random_forest": "Random Forest", "svm": "SVM (RBF)",
    "logistic_regression": "Logistic Regression", "extra_trees": "Extra Trees", "xgboost": "XGBoost",
}
alternative_predictors = {}
model_errors = {}
for model_id, factory, env_name in [
    ("svm", SVMPredictor, "ECG_SVM_MODEL_PATH"),
    ("logistic_regression", LogisticRegressionPredictor, "ECG_LOGISTIC_MODEL_PATH"),
    ("extra_trees", ExtraTreesPredictor, "ECG_EXTRA_TREES_MODEL_PATH"),
    ("xgboost", XGBoostPredictor, "ECG_XGBOOST_MODEL_PATH"),
]:
    try:
        alternative_predictors[model_id] = factory(model_path=os.environ.get(env_name))
    except Exception as exc:
        if os.environ.get(env_name):
            model_errors[model_id] = "Could not load the configured model artifact."
        elif model_id == "xgboost":
            model_errors[model_id] = "Optional XGBoost runtime is unavailable (libomp on this Mac)."
        else:
            model_errors[model_id] = "Model initialization failed; check the local runtime."


@app.get("/api/models")
async def list_models():
    return {"models": [
        {"id": model_id, "name": name, "available": model_id not in model_errors,
         "reason": model_errors.get(model_id),
         "is_demo": (predictor if model_id == "random_forest" else alternative_predictors.get(model_id)).is_demo if model_id not in model_errors else None}
        for model_id, name in MODEL_NAMES.items()
    ]}


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

from typing import List
from ingestion.wfdb_ingestor import WFDBECGIngestor
from ingestion.edf_ingestor import EDFECGIngestor

from fastapi.responses import StreamingResponse
import json

@app.post("/api/analyze")
async def analyze_ecg(files: List[UploadFile] = File(...), sampling_rate: float = 250.0, model_name: str = "random_forest", compare_models: bool = False):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    if model_name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail="Unknown model. Choose a model from the selector.")
    if model_name in model_errors:
        raise HTTPException(status_code=400, detail=model_errors[model_name])
    selected_predictor = predictor if model_name == "random_forest" else alternative_predictors[model_name]
    import uuid
    session_id = str(uuid.uuid4())
    session_dir = f"data/raw/{session_id}"
    os.makedirs(session_dir, exist_ok=True)
    temp_paths = []
    
    async def process_and_stream():
        try:
            yield json.dumps({"status": "progress", "progress": 5, "message": "Saving uploaded files..."}) + "\n"
            
            for file in files:
                path = os.path.join(session_dir, os.path.basename(file.filename))
                with open(path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                temp_paths.append(path)
                
            ingestor = None
            target_path = None
            
            edf_files = [p for p in temp_paths if p.lower().endswith('.edf')]
            if edf_files:
                ingestor = EDFECGIngestor()
                target_path = edf_files[0]
            elif any(p.lower().endswith('.hea') for p in temp_paths):
                hea_file = next(p for p in temp_paths if p.lower().endswith('.hea'))
                target_path = os.path.splitext(hea_file)[0]
                ingestor = WFDBECGIngestor()
            elif any(p.lower().endswith('.csv') for p in temp_paths):
                csv_file = next(p for p in temp_paths if p.lower().endswith('.csv'))
                ingestor = CSVECGIngestor(sampling_rate=sampling_rate)
                target_path = csv_file
                
            if not ingestor:
                yield json.dumps({"status": "error", "message": "Unsupported file format."}) + "\n"
                return

            # Execute pipeline steps manually to emit progress
            yield json.dumps({"status": "progress", "progress": 15, "message": "Ingesting signal data..."}) + "\n"
            raw_signal = ingestor.ingest(target_path)
            
            yield json.dumps({"status": "progress", "progress": 30, "message": f"Filtering & R-Peak detection ({raw_signal.duration_seconds:.1f}s signal)..."}) + "\n"
            processed_signal = preprocessor.preprocess(raw_signal)
            
            yield json.dumps({"status": "progress", "progress": 60, "message": "Extracting HRV metrics (This can take a while for long signals)..."}) + "\n"
            hrv_metrics = hrv_extractor.extract_features(processed_signal)
            
            import time
            from reporting.schemas import ModelComparisonResult
            comparisons = []
            model_ids = list(MODEL_NAMES) if compare_models else [model_name]
            for index, current_id in enumerate(model_ids):
                current_name = MODEL_NAMES[current_id]
                yield json.dumps({"status": "progress", "progress": 70 + int(27 * index / len(model_ids)), "message": f"Analyzing with {current_name}..."}) + "\n"
                if current_id in model_errors:
                    comparisons.append(ModelComparisonResult(model_id=current_id, model_name=current_name, error=model_errors[current_id]))
                    continue
                current_predictor = predictor if current_id == "random_forest" else alternative_predictors[current_id]
                try:
                    included_training = not hasattr(current_predictor.get_model(), "classes_")
                    started = time.perf_counter()
                    current_prediction = current_predictor.predict(hrv_metrics.feature_vector)
                    prediction_ms = (time.perf_counter() - started) * 1000
                    current_explainer = SHAPEngine(current_predictor.get_model(), current_predictor.background_data)
                    started = time.perf_counter()
                    current_prediction.top_shap_features = current_explainer.explain(current_predictor.prepare_features(hrv_metrics.feature_vector), top_k=5)
                    explanation_ms = (time.perf_counter() - started) * 1000
                    comparisons.append(ModelComparisonResult(
                        model_id=current_id, model_name=current_name, model_used=current_predictor.__class__.__name__, prediction=current_prediction,
                        prediction_time_ms=round(prediction_ms, 2), explanation_time_ms=round(explanation_ms, 2),
                        explanation_method=current_explainer.method, included_training=included_training,
                    ))
                except Exception as exc:
                    comparisons.append(ModelComparisonResult(model_id=current_id, model_name=current_name, error=str(exc)))
            selected = next(item for item in comparisons if item.model_id == model_name)
            if selected.prediction is None:
                raise ValueError(f"{selected.model_name}: {selected.error}")
            prediction_result = selected.prediction

            key_metrics = {
                "SDNN": hrv_metrics.time_domain.get("SDNN", 0.0),
                "RMSSD": hrv_metrics.time_domain.get("RMSSD", 0.0),
                "LF_HF_Ratio": hrv_metrics.frequency_domain.get("LF_HF_Ratio", 0.0),
                "SampEn": hrv_metrics.nonlinear.get("SampEn", 0.0)
            }
            
            # Construct final report
            from reporting.schemas import ECGStressReport
            import datetime
            report = ECGStressReport(
                report_id=f"REP-{uuid.uuid4().hex[:8].upper()}",
                timestamp=datetime.datetime.now().isoformat(),
                source_file=os.path.basename(target_path),
                prediction=prediction_result,
                model_comparison=comparisons if compare_models else [],
                key_hrv_summary=key_metrics,
                signal_analytics=build_signal_analytics(processed_signal),
                pipeline_metadata={
                    "duration_s": raw_signal.duration_seconds,
                    "sampling_rate_hz": raw_signal.sampling_rate,
                    "r_peaks_detected": len(processed_signal.r_peaks),
                    "model_used": selected_predictor.__class__.__name__,
                    "model_id": model_name,
                    "model_name": MODEL_NAMES[model_name],
                    "explanation_method": selected.explanation_method,
                    "prediction_time_ms": selected.prediction_time_ms,
                    "explanation_time_ms": selected.explanation_time_ms,
                    "comparison_requested": compare_models,
                    "included_training": selected.included_training,
                    "is_demo": prediction_result.is_demo,
                    "warnings": hrv_metrics.warnings
                }
            )
            
            yield json.dumps({"status": "complete", "data": report.model_dump(mode='json')}) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)

    return StreamingResponse(process_and_stream(), media_type="application/x-ndjson")

@app.post("/api/report.pdf")
async def download_pdf_report(report: ECGStressReport):
    from fastapi.responses import Response
    from reporting.pdf_reporter import PDFReportGenerator
    filename = "".join(char for char in report.report_id if char.isalnum() or char in "-_")[:64] or "ECG-report"
    return Response(PDFReportGenerator.generate(report), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
