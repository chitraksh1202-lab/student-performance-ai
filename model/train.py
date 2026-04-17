"""
Model training, feature engineering, and prediction logic.

Trains two models (Linear Regression + Random Forest) on a synthetic dataset,
compares their performance, and uses the better one for predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from data.generator import generate_dataset


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "consistency",
    "focus_efficiency",
    "improvement",
    "revision_strength",
    "distraction_index",
    "subject_strength",
]

# Weights for the readiness score formula (must sum to 1.0 across positive terms)
READINESS_WEIGHTS = {
    "consistency":      0.25,
    "focus_efficiency": 0.25,
    "revision_strength":0.20,
    "improvement":      0.15,
    "subject_strength": 0.10,
    # distraction_index is subtracted (penalty)
    "distraction_index":-0.15,
}


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(
    prev_marks: list,      # [mark1, mark2, mark3] — oldest to latest
    daily_hours: list,     # 7 daily study hours
    focused_time: float,   # avg focused hours/day
    revision_freq: int,    # sessions per week
    distraction: float,    # 0–10 scale
    subject_strength: float,  # 1–10 scale
) -> dict:
    """
    Convert raw user inputs into normalized [0,1] engineered features.

    Returns a dict with keys matching FEATURE_COLS.
    """
    hours = np.array(daily_hours, dtype=float)
    avg_hours = hours.mean()
    hour_std  = hours.std()

    # Consistency: how regularly the student studies (less variation = better)
    consistency = float(np.clip(1.0 - (hour_std / (avg_hours + 1e-6)), 0.0, 1.0))

    # Focus efficiency: focused hours as fraction of total study time
    focus_efficiency = float(np.clip(focused_time / (avg_hours + 1e-6), 0.0, 1.0))

    # Improvement slope via linear regression on 3 test marks
    # np.polyfit returns [slope, intercept]; slope = marks per test interval
    slope = float(np.polyfit([0, 1, 2], prev_marks, 1)[0])
    # Normalize: slope range roughly [-25, +25] → [0, 1]
    improvement = float(np.clip((slope + 25.0) / 50.0, 0.0, 1.0))

    # Revision strength: ordinal scale → continuous 0-1
    if revision_freq >= 5:
        revision_strength = 1.0
    elif revision_freq >= 3:
        revision_strength = 0.7
    elif revision_freq >= 1:
        revision_strength = 0.4
    else:
        revision_strength = 0.1

    distraction_index = distraction / 10.0
    subject_norm = (subject_strength - 1.0) / 9.0

    return {
        "consistency":       consistency,
        "focus_efficiency":  focus_efficiency,
        "improvement":       improvement,
        "revision_strength": revision_strength,
        "distraction_index": distraction_index,
        "subject_strength":  subject_norm,
    }


def compute_readiness(features: dict) -> tuple[float, dict]:
    """
    Compute readiness score (0–100) from features using a transparent
    weighted formula. Returns (score, breakdown_dict).

    breakdown_dict maps each feature to its signed contribution (0–100 scale).
    """
    breakdown = {}
    total = 0.0

    for feat, weight in READINESS_WEIGHTS.items():
        contribution = weight * features[feat] * 100
        breakdown[feat] = round(contribution, 1)
        total += contribution

    score = float(np.clip(total, 0, 100))
    return round(score, 1), breakdown


# ── Model bundle ───────────────────────────────────────────────────────────────

class PerformanceModel:
    """
    Trains and compares Linear Regression vs Random Forest.
    Uses MinMaxScaler for normalization (features are already 0-1 but
    scaling still helps LR with gradient sensitivity).
    Exposes the better model for predictions.
    """

    def __init__(self):
        self.lr  = LinearRegression()
        self.rf  = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        self.is_trained = False

        # Metrics
        self.lr_r2  = None;  self.lr_mae  = None
        self.rf_r2  = None;  self.rf_mae  = None
        self.best_model_name = None

        # For feature importance chart
        self.rf_importances: dict = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        """Train both models on synthetic data and pick the better one."""
        df = generate_dataset(n_samples=2000)

        X = df[FEATURE_COLS].values
        y = df["marks"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale (fit on train only)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        # Linear Regression
        self.lr.fit(X_train_s, y_train)
        lr_pred = self.lr.predict(X_test_s)
        self.lr_r2  = float(r2_score(y_test, lr_pred))
        self.lr_mae = float(mean_absolute_error(y_test, lr_pred))

        # Random Forest (doesn't need scaling, but we pass scaled for consistency)
        self.rf.fit(X_train_s, y_train)
        rf_pred = self.rf.predict(X_test_s)
        self.rf_r2  = float(r2_score(y_test, rf_pred))
        self.rf_mae = float(mean_absolute_error(y_test, rf_pred))

        # Feature importances from RF
        self.rf_importances = {
            col: round(float(imp), 4)
            for col, imp in zip(FEATURE_COLS, self.rf.feature_importances_)
        }

        # Pick best model by R²
        if self.rf_r2 >= self.lr_r2:
            self.best_model_name = "Random Forest"
            self._best = self.rf
            self.r2  = self.rf_r2
            self.mae = self.rf_mae
        else:
            self.best_model_name = "Linear Regression"
            self._best = self.lr
            self.r2  = self.lr_r2
            self.mae = self.lr_mae

        self.is_trained = True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict:
        """
        Predict exam marks and readiness score from engineered features.

        Returns:
            predicted_marks    — point estimate
            confidence_low/high — ±1.5 × MAE band (more realistic than ±MAE)
            readiness_score    — weighted 0-100 score
            readiness_breakdown — per-feature contributions
        """
        if not self.is_trained:
            self.train()

        X = np.array([[features[c] for c in FEATURE_COLS]])
        X_s = self.scaler.transform(X)
        pred = float(np.clip(self._best.predict(X_s)[0], 0, 100))

        # Confidence band: ±1.5×MAE gives a 90%+ coverage interval
        margin = 1.5 * (self.mae or 6.0)
        low  = round(max(0.0,   pred - margin), 1)
        high = round(min(100.0, pred + margin), 1)

        readiness, breakdown = compute_readiness(features)

        return {
            "predicted_marks":    round(pred, 1),
            "confidence_low":     low,
            "confidence_high":    high,
            "readiness_score":    readiness,
            "readiness_breakdown":breakdown,
            "features":           features,
        }

    # ── Model comparison summary ───────────────────────────────────────────────

    def comparison_summary(self) -> dict:
        """Return metrics for both models as a dict (for UI charts)."""
        return {
            "Linear Regression": {"R²": self.lr_r2, "MAE": self.lr_mae},
            "Random Forest":     {"R²": self.rf_r2, "MAE": self.rf_mae},
        }
