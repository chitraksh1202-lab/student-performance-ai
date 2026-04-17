"""
Synthetic dataset generator for the Student Performance Analyzer.

Generates 2000 realistic student behavioral records with engineered features
and a weighted exam mark target. All features are normalized to [0, 1].
"""

import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset of student behavioral metrics.

    Returns a DataFrame with columns:
        consistency, focus_efficiency, improvement, revision_strength,
        distraction_index, subject_strength  → all in [0, 1]
        marks → target variable [20, 100]
    """
    rng = np.random.default_rng(random_state)

    # ── Raw inputs ────────────────────────────────────────────────────────────
    avg_daily_hours  = rng.uniform(0.5, 8.0, n_samples)
    # Simulated day-to-day variation per student (lower = more consistent)
    hour_std         = rng.uniform(0.1, avg_daily_hours * 0.6)

    focused_time     = rng.uniform(0.2, avg_daily_hours)   # must be ≤ total
    revision_freq    = rng.integers(0, 9, n_samples)        # sessions/week
    distraction      = rng.uniform(0, 10, n_samples)
    subject_strength = rng.uniform(1, 10, n_samples)

    # Three successive test marks (oldest → latest)
    mark1 = rng.uniform(30, 80, n_samples)
    mark2 = np.clip(mark1 + rng.uniform(-10, 18, n_samples), 0, 100)
    mark3 = np.clip(mark2 + rng.uniform(-10, 18, n_samples), 0, 100)

    # ── Feature engineering ───────────────────────────────────────────────────

    # Consistency: lower std-dev relative to mean → more consistent [0,1]
    # Using coefficient of variation (std/mean), inverted and clipped
    consistency = np.clip(1.0 - (hour_std / (avg_daily_hours + 1e-6)), 0.0, 1.0)

    # Focus efficiency: proportion of study time that was focused [0,1]
    focus_efficiency = np.clip(focused_time / (avg_daily_hours + 1e-6), 0.0, 1.0)

    # Improvement slope: linear regression slope of 3 marks, normalized
    # For 3 evenly-spaced points, slope = (m3 - m1) / 2  (vectorized polyfit)
    slope = (mark3 - mark1) / 2.0          # marks per test interval
    # Normalize: typical range [-25, +25] → [0, 1]
    improvement = np.clip((slope + 25.0) / 50.0, 0.0, 1.0)

    # Revision strength: ordinal → continuous [0,1]
    revision_strength = np.where(
        revision_freq >= 5, 1.0,
        np.where(revision_freq >= 3, 0.7,
        np.where(revision_freq >= 1, 0.4, 0.1))
    )

    # Distraction index: higher = more distracted [0,1]
    distraction_index = distraction / 10.0

    # Subject strength normalized to [0,1]
    subject_norm = (subject_strength - 1.0) / 9.0

    # ── Target variable ───────────────────────────────────────────────────────
    # Weighted formula that reflects real-world relationships
    marks = (
        18.0 * consistency
        + 22.0 * focus_efficiency
        + 12.0 * improvement          # improvement is already 0-1
        + 14.0 * revision_strength
        + 8.0  * subject_norm
        - 14.0 * distraction_index
        + 0.28 * mark3                # recent performance carries weight
        + rng.normal(0, 4, n_samples) # realistic noise
    )
    marks = np.clip(marks, 20, 100)

    return pd.DataFrame({
        "consistency":      consistency,
        "focus_efficiency": focus_efficiency,
        "improvement":      improvement,
        "revision_strength":revision_strength,
        "distraction_index":distraction_index,
        "subject_strength": subject_norm,
        "marks":            marks,
    })
