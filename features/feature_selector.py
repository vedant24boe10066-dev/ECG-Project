import numpy as np
import pandas as pd
from typing import List
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

class HRVFeatureSelector:
    """
    Applies statistical selection to isolate the most discriminative HRV features.
    """
    
    def __init__(self, method: str = 'mutual_info', k: int = 5):
        """
        :param method: 'mutual_info' or 'anova'
        :param k: Number of top features to select
        """
        self.method = method
        self.k = k
        self.selector = None
        self.selected_feature_names = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fits the feature selector on training data."""
        score_func = mutual_info_classif if self.method == 'mutual_info' else f_classif
        self.selector = SelectKBest(score_func=score_func, k=min(self.k, X.shape[1]))
        self.selector.fit(X, y)
        
        # Get mask of selected features
        mask = self.selector.get_support()
        self.selected_feature_names = X.columns[mask].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reduces the dataset to the selected features."""
        if self.selected_feature_names is None:
            raise ValueError("Selector must be fitted before calling transform.")
        return X[self.selected_feature_names]

    def get_selected_features(self) -> List[str]:
        return self.selected_feature_names if self.selected_feature_names else []
