"""
train_model.py

Trains the final XGBoost model for the House Price project and persists
everything an inference file needs later:
  - the fitted cleaning pipeline (categories learned on train must be
    reused as-is on test — never refit on test, see housing_cleaning.py)
  - the final trained XGBRegressor

This is steps 1-7 from the original run_training.py, split out on its own
so training and prediction each live in one focused file. The Stratified
K-Fold CV below is only for a trustworthy performance estimate — none of
the fold models are the one that gets saved; the model saved at the end
is refit on ALL of X (step 7), same as before.
"""

import os

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from Data_Loader import DataLoader
from housing_cleaning import (
    build_cleaning_pipeline,
    remove_training_outliers,
    transform_target,
    inverse_transform_target,
)
from config import (
    TRAIN_PATH,
    TEST_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PIPELINE_PATH,
    TARGET_COL,
    RANDOM_STATE,
    CV_N_SPLITS,
    CV_QCUT_BINS,
    XGB_PARAMS,
    TOP_NUMERICAL_COLUMNS
)


def main():
    # 1. Load train data only — this file's job is training, not prediction.
    #    (DataLoader still needs both paths in its constructor, but we
    #    never touch test_path here.)
    loader = DataLoader(TRAIN_PATH, TEST_PATH)
    train_df = loader.load_train_data()

    # 2. Remove training-only outlier rows — must happen BEFORE the target
    #    log transform (see housing_cleaning.py docstring)
    train_df = remove_training_outliers(train_df)
    train_df = train_df.reset_index(drop=True)

    # 3. Log-transform the target
    train_df[TARGET_COL] = transform_target(train_df[TARGET_COL])

    # 4. Split X / y
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]

    # 5. Fit the cleaning pipeline on train ONLY. This exact fitted object
    #    gets saved and reused as-is at inference time.
    cleaning_pipeline = build_cleaning_pipeline()
    X = cleaning_pipeline.fit_transform(X)

    # 6. Stratified K-Fold CV (target binned via qcut) — trustworthy performance estimate only.
    price_bins = pd.qcut(y, q=CV_QCUT_BINS, labels=False)
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    cv_mae_scores = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, price_bins), start=1):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        fold_model = XGBRegressor(**XGB_PARAMS)
        fold_model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], verbose=False)

        preds = fold_model.predict(X_valid_fold)
        real_preds = inverse_transform_target(preds)
        real_y_valid = inverse_transform_target(y_valid_fold)

        fold_mae = mean_absolute_error(real_y_valid, real_preds)
        cv_mae_scores.append(fold_mae)
        print(f"Fold {fold} Real MAE: ${fold_mae:,.2f}")

    print("-" * 35)
    print(f"Mean Stratified CV MAE: ${np.mean(cv_mae_scores):,.2f}")

    # 7. Refit ONE final model on ALL of X — this is the model that gets saved.

    final_params = XGB_PARAMS.copy()
    final_params.pop("early_stopping_rounds", None)

    final_model = XGBRegressor(**final_params)
    final_model.fit(X, y)

    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": final_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)
    print("\nTop 15 features:")
    print(feature_importances.head(TOP_NUMERICAL_COLUMNS))

    # 8. Persist both fitted artifacts for predict.py to reuse
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(cleaning_pipeline, PIPELINE_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved cleaning pipeline to: {PIPELINE_PATH}")


if __name__ == "__main__":
    main()