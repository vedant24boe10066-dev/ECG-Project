import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import cross_validate
from models.classifiers import RandomForestPredictor, SVMPredictor, XGBoostPredictor

class ModelBenchmarker:
    """Evaluates and compares RF, SVM, and XGBoost models using cross-validation."""
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestPredictor(),
            'SVM': SVMPredictor(),
            'XGBoost': XGBoostPredictor()
        }
        
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, cv: int = 5) -> pd.DataFrame:
        results = []
        scoring = ['accuracy', 'f1', 'roc_auc']
        
        for name, predictor in self.models.items():
            model = predictor.get_model()
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
            
            results.append({
                'Model': name,
                'Accuracy': np.mean(cv_results['test_accuracy']),
                'F1-Score': np.mean(cv_results['test_f1']),
                'ROC-AUC': np.mean(cv_results['test_roc_auc'])
            })
            
        df_results = pd.DataFrame(results)
        return df_results.sort_values(by='F1-Score', ascending=False)
