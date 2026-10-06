import numpy as np
import pandas as pd
from typing import Dict, Any
from typing import Dict, Any, Optional
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from models.base import BaseStressPredictor
from reporting.schemas import StressPredictionResult

def generate_demo_training(feature_names: list[str], random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Common synthetic teaching dataset; labels are not real stress annotations."""
    rng = np.random.default_rng(random_state)
    # Synthetic teaching scenarios in HRV units, not random labels on 0–1 values.
    # This rule illustrates lower variability/higher rate; it is not a clinical threshold.
    count = 600
    rmssd = rng.uniform(4, 85, count)
    sdnn = np.clip(rmssd * rng.uniform(0.7, 1.3, count), 4, 110)
    mean_rr = rng.uniform(550, 1100, count)
    demo_columns = {
        "RMSSD": rmssd, "SDNN": sdnn, "MeanRR": mean_rr,
        "pNN50": np.clip((rmssd - 15) * 1.2, 0, 100),
        "SD1": rmssd / np.sqrt(2),
        "SD2": np.sqrt(np.maximum(2 * sdnn ** 2 - rmssd ** 2 / 2, 1)),
        "LF": rng.uniform(0, 2, count), "HF": rng.uniform(0, 2, count),
        "LF_HF_Ratio": rng.uniform(0.1, 5, count), "SampEn": rng.uniform(0.5, 2.5, count),
    }
    X_demo = pd.DataFrame({name: demo_columns.get(name, rng.random(count)) for name in feature_names})
    score = (25 - rmssd) / 12 + (20 - sdnn) / 16 + (800 - mean_rr) / 500
    # Soft synthetic labels and larger leaves avoid claiming certainty in a toy model.
    illustrative_probability = 0.08 + 0.84 / (1 + np.exp(-score))
    return X_demo, (rng.random(count) < illustrative_probability).astype(int)


class RandomForestPredictor(BaseStressPredictor):
    """Explicit, reproducible demo mode or a locally trained model."""

    def __init__(self, model_path: Optional[str] = None, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.feature_names = []
        self.is_demo = True
        self.background_data = None
        if model_path:
            artifact = joblib.load(model_path)  # Missing files must fail, not fall back to a demo.
            if isinstance(artifact, dict):
                self.model = artifact["model"]
                self.feature_names = artifact["feature_names"]
                self.is_demo = artifact.get("is_demo", False)
                self.background_data = artifact.get("background_data")
            else:
                self.model = artifact
                self.feature_names = list(getattr(self.model, "feature_names_in_", []))
                self.is_demo = False
            if not self.feature_names:
                raise ValueError("Saved model needs its training feature names. Save it using save_model().")
            self._validate_classes()

    def _validate_classes(self):
        if set(self.model.classes_) != {0, 1}:
            raise ValueError("Training labels must contain both 0 (No-Stress) and 1 (Stress).")

    def prepare_features(self, features):
        names = self.feature_names or list(features)
        if set(features) != set(names):
            raise ValueError("HRV features do not match the model's training feature names.")
        unavailable = [name for name in names if features[name] is None or not np.isfinite(features[name])]
        if unavailable and not self.is_demo:
            raise ValueError("Cannot classify: unavailable HRV features: " + ", ".join(unavailable))
        # Demo-only placeholders; the measured HRV report retains nulls and warnings.
        return {name: 0.0 if name in unavailable else float(features[name]) for name in names}

    def predict(self, features: Dict[str, float]) -> StressPredictionResult:
        prepared = self.prepare_features(features)
        if not hasattr(self.model, "classes_"):
            X_demo, y_demo = generate_demo_training(list(prepared), self.random_state)
            if "min_samples_leaf" in self.model.get_params(deep=False):
                self.model.set_params(min_samples_leaf=8)
            self.fit(X_demo, y_demo, demo=True)
        df = pd.DataFrame([prepared])[self.feature_names]
        stress_index = list(self.model.classes_).index(1)
        stress_prob = float(self.model.predict_proba(df)[0][stress_index])
        return StressPredictionResult(
            prediction_label="Stress" if stress_prob > 0.5 else "No-Stress",
            stress_probability=stress_prob,
            confidence_score=max(stress_prob, 1 - stress_prob),
            top_shap_features=[],
            is_demo=self.is_demo,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, demo: bool = False) -> None:
        if X.empty or not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Training features must be nonempty and finite.")
        if set(np.asarray(y)) != {0, 1}:
            raise ValueError("Training labels must contain both 0 (No-Stress) and 1 (Stress).")
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        self.is_demo = demo
        self.background_data = X.sample(n=min(24, len(X)), random_state=self.random_state).copy()

    def save_model(self, path: str) -> None:
        self._validate_classes()
        joblib.dump({"model": self.model, "feature_names": self.feature_names, "is_demo": self.is_demo, "background_data": self.background_data}, path)

    def get_model(self) -> Any:
        return self.model


class SVMPredictor(RandomForestPredictor):
    """Scaled RBF SVM using the same feature schema and demo scenarios as RF."""

    def __init__(self, random_state: int = 42, model_path: Optional[str] = None):
        super().__init__(model_path=model_path, random_state=random_state)
        if not model_path:
            self.model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=random_state))


class LogisticRegressionPredictor(RandomForestPredictor):
    def __init__(self, random_state: int = 42, model_path: Optional[str] = None):
        super().__init__(model_path=model_path, random_state=random_state)
        if not model_path:
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state))


class ExtraTreesPredictor(RandomForestPredictor):
    def __init__(self, random_state: int = 42, model_path: Optional[str] = None):
        super().__init__(model_path=model_path, random_state=random_state)
        if not model_path:
            self.model = ExtraTreesClassifier(n_estimators=100, random_state=random_state)


class XGBoostPredictor(RandomForestPredictor):
    def __init__(self, random_state: int = 42, model_path: Optional[str] = None):
        super().__init__(model_path=model_path, random_state=random_state)
        if not model_path:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric="logloss")
