#!/usr/bin/env python3
"""
forecast_models.py

Step 4 Classical Time Series Forecasting Models for India Urban-Change:
Implements 4 parsimonious models suitable for short annual series (N=9, 2017-2025):
1. Naive Persistence Baseline (last observed value)
2. Drift Baseline (random walk with historical slope)
3. Constrained ARIMA (ARIMA(0,1,0) with drift or ARIMA(1,1,0))
4. Holt's Linear Exponential Smoothing (level + additive trend, non-seasonal)

Methodological Guardrails:
- Chronological ordering strictly preserved; zero shuffling.
- Rejects seasonal Holt-Winters (annual data has period m=1, no seasonal cycle).
- Constrains parameter count to prevent overfitting on N_train=7 observations.
- Handles short or constant series gracefully.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import Holt
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class NaivePersistenceForecaster:
    """Model 1: Naive Persistence Baseline (y_{T+h} = y_T)."""
    def __init__(self):
        self.last_val = None

    def fit(self, y_train: np.ndarray):
        self.last_val = float(y_train[-1])
        return self

    def predict(self, steps: int = 2) -> np.ndarray:
        return np.full(steps, self.last_val, dtype=float)


class DriftForecaster:
    """Model 2: Drift Baseline (Random walk with linear historical slope)."""
    def __init__(self):
        self.last_val = None
        self.slope = 0.0

    def fit(self, y_train: np.ndarray):
        n = len(y_train)
        self.last_val = float(y_train[-1])
        if n > 1:
            self.slope = float((y_train[-1] - y_train[0]) / (n - 1))
        else:
            self.slope = 0.0
        return self

    def predict(self, steps: int = 2) -> np.ndarray:
        h = np.arange(1, steps + 1)
        preds = self.last_val + h * self.slope
        # Built fraction cannot be negative
        return np.maximum(preds, 0.0)


class HoltLinearForecaster:
    """Model 3: Holt's Linear Exponential Smoothing (Level + Trend)."""
    def __init__(self, alpha: float = 0.5, beta: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None

    def fit(self, y_train: np.ndarray):
        y = np.asarray(y_train, dtype=float)
        n = len(y)
        if np.all(y == y[0]) or np.var(y) < 1e-12:
            self.level = y[-1]
            self.trend = 0.0
            return self

        if HAS_STATSMODELS and n >= 4:
            try:
                model = Holt(y, initialization_method="estimated").fit(optimized=True)
                self.sm_model = model
                return self
            except Exception:
                self.sm_model = None

        # Fallback recursive Holt formulas
        self.sm_model = None
        self.level = y[0]
        self.trend = y[1] - y[0] if n > 1 else 0.0
        for t in range(1, n):
            prev_level = self.level
            self.level = self.alpha * y[t] + (1 - self.alpha) * (prev_level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        return self

    def predict(self, steps: int = 2) -> np.ndarray:
        if getattr(self, "sm_model", None) is not None:
            preds = self.sm_model.forecast(steps)
            return np.maximum(np.asarray(preds, dtype=float), 0.0)
        h = np.arange(1, steps + 1)
        preds = self.level + h * self.trend
        return np.maximum(preds, 0.0)


class ARIMAForecaster:
    """
    Model 4: Parsimonious ARIMA for short series.
    Default order: (0, 1, 0) with drift (random walk with drift on differences).
    """
    def __init__(self, order: Tuple[int, int, int] = (0, 1, 0), trend: str = "t"):
        self.order = order
        self.trend = trend
        self.sm_fit = None
        self.fallback_drift = None

    def fit(self, y_train: np.ndarray):
        y = np.asarray(y_train, dtype=float)
        if np.all(y == y[0]) or np.var(y) < 1e-12:
            self.fallback_drift = DriftForecaster().fit(y)
            return self

        if HAS_STATSMODELS:
            try:
                # Constrained ARIMA(0,1,0) or (1,1,0)
                model = ARIMA(y, order=self.order, trend=self.trend)
                self.sm_fit = model.fit()
                return self
            except Exception:
                self.sm_fit = None

        # Fallback to DriftForecaster if statsmodels unavailable or fails to converge
        self.fallback_drift = DriftForecaster().fit(y)
        return self

    def predict(self, steps: int = 2) -> np.ndarray:
        if self.sm_fit is not None:
            try:
                preds = self.sm_fit.forecast(steps=steps)
                return np.maximum(np.asarray(preds, dtype=float), 0.0)
            except Exception:
                pass
        return self.fallback_drift.predict(steps=steps)
