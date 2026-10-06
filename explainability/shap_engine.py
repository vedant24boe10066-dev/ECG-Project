import shap
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from reporting.schemas import SHAPAttribution


class SHAPEngine:
    """Explain the Stress-class probability with tree or model-agnostic SHAP."""

    def __init__(self, model: Any = None, background_data: pd.DataFrame | None = None):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self.method = None

    def set_model(self, model: Any, background_data: pd.DataFrame | None = None) -> None:
        if model is not self.model or background_data is not self.background_data:
            self.model = model
            self.background_data = background_data
            self.explainer = None
            self.method = None

    def _build_probability_explainer(self, names):
        if self.background_data is None:
            raise ValueError("This model needs training background data for SHAP. Save it using save_model().")
        stress_index = list(self.model.classes_).index(1)

        def stress_probability(values):
            return self.model.predict_proba(pd.DataFrame(values, columns=names))[:, stress_index]

        self.explainer = shap.KernelExplainer(stress_probability, self.background_data[names])
        self.method = "KernelSHAP"

    def explain(self, feature_vector: Dict[str, float], top_k: int = 5) -> List[SHAPAttribution]:
        if self.model is None:
            return []
        df = pd.DataFrame([feature_vector])
        if hasattr(self.model, "feature_names_in_"):
            df = df[list(self.model.feature_names_in_)]
        is_tree = isinstance(self.model, (RandomForestClassifier, ExtraTreesClassifier)) or self.model.__class__.__module__.startswith('xgboost')
        if self.explainer is None:
            if is_tree:
                if self.model.__class__.__module__.startswith('xgboost'):
                    if self.background_data is None:
                        raise ValueError("XGBoost needs training background data to explain probability output.")
                    try:
                        self.explainer = shap.TreeExplainer(self.model, data=self.background_data[df.columns], model_output="probability", feature_perturbation="interventional")
                        self.method = "TreeSHAP"
                    except Exception:
                        # Some XGBoost/SHAP combinations cannot load trees for probability
                        # explanations. Explain the same predict_proba output instead.
                        self._build_probability_explainer(df.columns.tolist())
                else:
                    self.explainer = shap.TreeExplainer(self.model)
                    self.method = "TreeSHAP"
            else:
                self._build_probability_explainer(df.columns.tolist())
        if self.method == "TreeSHAP":
            try:
                values = self.explainer.shap_values(df)
            except Exception:
                if not self.model.__class__.__module__.startswith('xgboost'):
                    raise
                self._build_probability_explainer(df.columns.tolist())
        if self.method == "TreeSHAP":
            stress_index = list(self.model.classes_).index(1)
            if isinstance(values, list):
                values = values[stress_index][0]
            elif values.ndim == 3:
                values = values[0, :, stress_index]
            else:
                values = values[0]
        else:
            # Enumerate the ten-feature demo's coalitions, rather than random sampling.
            values = self.explainer.shap_values(df, nsamples=2 ** min(len(df.columns), 10), silent=True, l1_reg=0.0)[0]
        attributions = [SHAPAttribution(
            feature_name=name, shap_value=float(values[i]), feature_value=float(df.iloc[0, i]),
            impact_direction="increases_stress" if values[i] > 0 else "decreases_stress",
        ) for i, name in enumerate(df.columns)]
        return sorted(attributions, key=lambda item: abs(item.shap_value), reverse=True)[:top_k]
