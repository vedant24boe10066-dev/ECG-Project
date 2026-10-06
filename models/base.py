from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import pandas as pd
from reporting.schemas import StressPredictionResult

class BaseStressPredictor(ABC):
    """Abstract base class for all ML classification models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Trains the model on the feature matrix."""
        pass
        
    @abstractmethod
    def predict(self, features: Dict[str, float]) -> StressPredictionResult:
        """Predicts stress class from a single feature vector."""
        pass
        
    @abstractmethod
    def get_model(self) -> Any:
        """Returns the underlying scikit-learn or XGBoost model (needed for SHAP)."""
        pass
