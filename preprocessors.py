# preprocessors.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ClinicalScaler(BaseEstimator, TransformerMixin):
    """
    Multiply specified clinical features by a fixed factor (default 0.1).
    Designed to work with pandas DataFrames.
    """
    def __init__(self, clinical_features, factor=0.1):
        self.clinical_features = clinical_features
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in self.clinical_features:
                X[col] = X[col].astype(float) * self.factor
            return X
        return X