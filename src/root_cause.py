"""Predict the likely laser-process fault from image-measured verification features."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURES = [
    "decoded", "cell_contrast", "axial_nonuniformity", "unused_ec", "module_agreement",
    "light_mean", "dark_mean", "light_cv", "dark_cv", "sharpness", "dot_growth",
]
GRADE_ONLY = ["decoded", "overall_rank"]
MISSING = -1.0


@dataclass
class RootCauseResults:
    """Metrics for the root-cause classifier and its baselines."""
    accuracy: float
    majority_baseline: float
    grade_only_baseline: float
    hidden_fault_recall: float
    n_hidden_faults: int
    labels: list[str]
    confusion: np.ndarray
    feature_importances: pd.Series


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["overall_rank"] = out["overall"].map("ABCDF".index)
    out[FEATURES] = out[FEATURES].astype(float).fillna(MISSING)
    return out


def train_root_cause(
    df: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42, n_estimators: int = 300,
) -> tuple[RandomForestClassifier, RootCauseResults]:
    df = prepare(df)
    tr, te = train_test_split(df, test_size=test_size, stratify=df["fault"], random_state=random_seed)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    model.fit(tr[FEATURES], tr["fault"])
    pred = model.predict(te[FEATURES])

    grade_only = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    grade_only.fit(tr[GRADE_ONLY], tr["fault"])

    hidden = te[(te["overall"] == "A") & (te["fault"] != "ok")]
    hidden_pred = model.predict(hidden[FEATURES]) if len(hidden) else np.array([])

    labels = sorted(df["fault"].unique())
    res = RootCauseResults(
        accuracy=accuracy_score(te["fault"], pred),
        majority_baseline=float(te["fault"].value_counts(normalize=True).max()),
        grade_only_baseline=accuracy_score(te["fault"], grade_only.predict(te[GRADE_ONLY])),
        hidden_fault_recall=float((hidden_pred != "ok").mean()) if len(hidden) else float("nan"),
        n_hidden_faults=len(hidden),
        labels=labels,
        confusion=confusion_matrix(te["fault"], pred, labels=labels),
        feature_importances=pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False),
    )
    logger.info("accuracy %.3f | grade-only %.3f | majority %.3f", res.accuracy, res.grade_only_baseline, res.majority_baseline)
    logger.info("faulty marks graded A in test: %d, flagged as faulty: %.1f%%", res.n_hidden_faults, 100 * res.hidden_fault_recall)
    return model, res
