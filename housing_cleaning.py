"""
housing_cleaning.py

Replaces the earlier "one class per column" design with:
  1. One central DECISIONS table — every column's cleaning strategy lives
     here, in one place, instead of being scattered across separate
     transformer classes.
  2. One generic DecisionBasedCleaner that reads the table and dispatches
     to a small set of reusable strategy handlers.
  3. A dynamic safety check, using DataInspector itself (not a
     reimplementation of its logic), that warns if a column with high
     missing % shows up with no decision in the table — e.g. if the data
     changes or a new feature appears later.

Why the table still names columns explicitly: each strategy here (drop /
binary / ordinal map) was chosen empirically by testing model performance,
not derived purely from the missing percentage — Alley, for example, has
a perfectly reasonable sample size but was dropped because it tested worse
than keeping it. There's no way to "auto-detect" an empirical result, so
the decision itself has to be written down — the goal here is only to
avoid repeating the SAME kind of logic (drop / binary / ordinal) in a new
class every time, and to never silently skip a column that needs a
decision.

Row-level operations (outlier removal, target log-transform) are still
kept OUTSIDE the pipeline, for the same reason as before: they are not
per-row fit/transform operations, and outlier removal must never touch
the test set.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

try:
    from Data_Inspector import DataInspector
except ImportError:
    DataInspector = None  # dynamic safety check is skipped if unavailable

from config import HIGH_MISSING_THRESHOLD


# ---------------------------------------------------------------------------
# Row-level operations (kept OUTSIDE the pipeline on purpose — see module
# docstring). Apply these once, manually, before building X/y.
# ---------------------------------------------------------------------------

def remove_training_outliers(train_df: pd.DataFrame) -> pd.DataFrame:
    """Drops the manually-identified bad outlier rows from the TRAINING data only.

    Never call this on the test set — every test row must still get a prediction.
    """
    df = train_df.copy()

    bad_idx_1 = df[
        ((df["GrLivArea"] > 4500) & (df["SalePrice"] < 200000))
        | ((df["GarageArea"] > 1200) & (df["SalePrice"] < 300000))
    ].index
    df = df.drop(index=bad_idx_1)

    bad_idx_2 = df[(df["TotalBsmtSF"] > 3000) & (df["1stFlrSF"] < 2500)].index
    df = df.drop(index=bad_idx_2)

    return df.reset_index(drop=True)


def transform_target(y: pd.Series) -> pd.Series:
    """log1p transform for SalePrice — fit the model on this scale."""
    return np.log1p(y)


def inverse_transform_target(y_pred) -> np.ndarray:
    """Turns model predictions back into real dollar values."""
    return np.expm1(y_pred)


# ---------------------------------------------------------------------------
# The central decision table. To add/change a column's treatment, edit
# ONE entry here — nothing else in this file needs to change.
# ---------------------------------------------------------------------------

DECISIONS = {
    "Id": {"strategy": "drop"},
    "PoolQC": {"strategy": "drop"},
    "Alley": {"strategy": "drop"},

    "MiscFeature": {
        "strategy": "binary_isin",
        "categories": ["Shed", "Gar2", "Othr", "TenC"],
        "drop_with": ["MiscVal"],
    },
    "Fence": {"strategy": "binary_notna"},

    "MasVnrType": {
        "strategy": "ordinal_map",
        "fillna_before": "None",
        "map": {"None": 0, "BrkCmn": 1, "CBlock": 1, "BrkFace": 1, "Stone": 1},
    },
    "FireplaceQu": {
        "strategy": "ordinal_map",
        "fillna_before": "None",
        "map": {"Ex": 1, "Gd": 1, "TA": 1, "Fa": 1, "Po": 1, "None": 0},
    },

    "MasVnrArea": {"strategy": "constant_impute", "value": 0},

    "GrLivArea": {"strategy": "log1p"},
}


# ---------------------------------------------------------------------------
# Strategy handlers — each one is a small, reusable function. Adding a new
# strategy means adding one function here + one entry in STRATEGY_HANDLERS,
# not a whole new class.
# ---------------------------------------------------------------------------

def _apply_drop(X, col, spec):
    return X.drop(columns=[col], errors="ignore")


def _apply_binary_isin(X, col, spec):
    X[col] = X[col].isin(spec["categories"]).astype(int)
    return X.drop(columns=spec.get("drop_with", []), errors="ignore")


def _apply_binary_notna(X, col, spec):
    X[col] = X[col].notna().astype(int)
    return X


def _apply_ordinal_map(X, col, spec):
    series = X[col]
    if "fillna_before" in spec:
        series = series.fillna(spec["fillna_before"])
    X[col] = series.map(spec["map"]).fillna(spec.get("map_fillna", 0)).astype(int)
    return X


def _apply_constant_impute(X, col, spec):
    X[col] = X[col].fillna(spec["value"])
    return X


def _apply_log1p(X, col, spec):
    X[col] = np.log1p(X[col])
    return X


STRATEGY_HANDLERS = {
    "drop": _apply_drop,
    "binary_isin": _apply_binary_isin,
    "binary_notna": _apply_binary_notna,
    "ordinal_map": _apply_ordinal_map,
    "constant_impute": _apply_constant_impute,
    "log1p": _apply_log1p,
}


class DecisionBasedCleaner(BaseEstimator, TransformerMixin):
    """Applies every column's strategy from a decisions table.

    On fit, if DataInspector is importable, it's used to dynamically check
    for any column with high missing % that has no entry in the table yet —
    reusing DataInspector's own missing-value logic instead of
    reimplementing it here. This never auto-decides anything; it only
    warns, so a newly-appeared high-missing column can't silently slip
    through unnoticed.
    """

    def __init__(self, decisions: dict = None, high_missing_threshold: float = HIGH_MISSING_THRESHOLD):
        self.decisions = decisions if decisions is not None else DECISIONS
        self.high_missing_threshold = high_missing_threshold

    def fit(self, X, y=None):
        if DataInspector is not None:
            inspector = DataInspector(X)
            high_missing = inspector.get_columns_contain_high_missing_values(
                threshold_percent=self.high_missing_threshold
            )
            undecided = [c for c in high_missing.index if c not in self.decisions]
            for col in undecided:
                warnings.warn(
                    f"Column '{col}' has high missing % but no decision in "
                    f"the DECISIONS table yet — it will be left untouched."
                )
        return self

    def transform(self, X):
        X = X.copy()
        for col, spec in self.decisions.items():
            if col not in X.columns:
                continue  # already dropped by an earlier entry, or not present
            handler = STRATEGY_HANDLERS[spec["strategy"]]
            X = handler(X, col, spec)
        return X


class CategoricalTypeAligner(BaseEstimator, TransformerMixin):
    """Converts every remaining object-dtype column to pandas 'category'
    dtype, for XGBoost's enable_categorical=True.

    Fitting on train and reusing those exact categories on test means a
    category value seen only in test becomes NaN on transform — treated as
    a missing value by XGBoost, consistent with leaving other missing
    columns unimputed on purpose.
    """

    def fit(self, X, y=None):
        self.cat_cols_ = list(X.select_dtypes(include=["object"]).columns)
        self.categories_ = {
            col: sorted(X[col].dropna().unique().tolist()) for col in self.cat_cols_
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cat_cols_:
            dtype = pd.CategoricalDtype(categories=self.categories_[col])
            X[col] = X[col].astype(dtype)
        return X


def build_cleaning_pipeline(decisions: dict = None) -> Pipeline:
    """Builds the full feature-cleaning pipeline.

    Fit this ONCE on the training features (after outlier rows have been
    removed and before the target is separated), then use the same fitted
    pipeline's .transform() on the test features — never re-fit on test.
    """
    return Pipeline(steps=[
        ("decision_cleaning", DecisionBasedCleaner(decisions=decisions)),
        ("categorical_alignment", CategoricalTypeAligner()),
    ])
