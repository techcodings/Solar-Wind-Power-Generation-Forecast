# src/models.py
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

class GBMPointModel:
    def __init__(self):
        # squared_error = L2
        self.model = HistGradientBoostingRegressor(loss="squared_error", max_depth=None, max_bins=255)
        self.feats = None

    def fit(self, X, y, feats=None):
        self.feats = feats
        self.model.fit(X, y)

    def forecast(self, Xf):
        return self.model.predict(Xf)

class QuantileGBM:
    def __init__(self, quantile: float):
        # HistGBR supports quantile loss with alpha
        self.q = quantile
        self.model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_depth=None, max_bins=255)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
