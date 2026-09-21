"""
config.py

Single source of truth for paths, hyperparameters, and thresholds used
across the project. Every other file should import from here instead of
repeating a literal value — if something needs to change (a path, the
target column, a hyperparameter), it changes in exactly one place.

Per-column cleaning DECISIONS in housing_cleaning.py stay where they are
on purpose: those are empirical, column-specific judgment calls (see that
file's docstring), not general project settings — this file is for the
things that are genuinely the same everywhere they're used.
"""

import os

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
TRAIN_PATH = r"C:\Users\Belal Moustafa\Desktop\Machine Learning\Kaggle\Housing Prices\train.csv"
TEST_PATH = r"C:\Users\Belal Moustafa\Desktop\Machine Learning\Kaggle\Housing Prices\test.csv"

# ---------------------------------------------------------------------------
# Saved-artifact paths (written by train_model.py, read by predict.py)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "final_model.joblib")
PIPELINE_PATH = os.path.join(MODEL_DIR, "cleaning_pipeline.joblib")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

# ---------------------------------------------------------------------------
# Target column — referenced across DataInspector calls, housing_cleaning,
# train_model.py and predict.py
# ---------------------------------------------------------------------------
TARGET_COL = "SalePrice"

# ---------------------------------------------------------------------------
# Reproducibility — one seed reused everywhere a random_state is needed
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# numerical mixin — number of columns used in the correlation matrix visulaization
# ---------------------------------------------------------------------------
TOP_NUMERICAL_COLUMNS = 15

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_N_SPLITS = 5
CV_QCUT_BINS = 5  # number of SalePrice bins used to stratify the folds

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
HIGH_MISSING_THRESHOLD = 40.0  # % — DecisionBasedCleaner's undecided-column check

# ---------------------------------------------------------------------------
# Model hyperparameters (XGBoost)
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=900,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=6,
    objective="reg:absoluteerror",
    early_stopping_rounds= 50
)
