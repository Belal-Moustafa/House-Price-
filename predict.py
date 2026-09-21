"""
predict.py

Loads the model and cleaning pipeline saved by train_model.py, generates
predictions for the Kaggle test set, and writes submission.csv in the
exact format this competition expects: two columns, Id and SalePrice,
no index column.

Run train_model.py first — this file only loads what it already saved.
"""

import joblib
import pandas as pd

from Data_Loader import DataLoader
from housing_cleaning import inverse_transform_target
from config import (
    TRAIN_PATH,
    TEST_PATH,
    MODEL_PATH,
    PIPELINE_PATH,
    SUBMISSION_PATH,
)


def main():
    # 1. Load test data only
    loader = DataLoader(TRAIN_PATH, TEST_PATH)
    test_df = loader.load_test_data()

    # 2. Keep the Id column aside BEFORE transforming — the cleaning
    #    pipeline drops "Id" (see housing_cleaning.DECISIONS), and Kaggle
    #    needs it back in the submission file
    test_ids = test_df["Id"].copy()

    # 3. Load the EXACT fitted pipeline used on train (same learned
    #    categories, same column decisions) — never refit on test
    cleaning_pipeline = joblib.load(PIPELINE_PATH)
    X_test = cleaning_pipeline.transform(test_df.copy())

    # 4. Load the trained model and predict (still on the log1p scale)
    final_model = joblib.load(MODEL_PATH)
    test_preds_log = final_model.predict(X_test)

    # 5. Back to real dollar values — inverse of transform_target's log1p
    test_preds = inverse_transform_target(test_preds_log)

    # 6. Kaggle's exact expected format for this competition
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": test_preds,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Saved submission file to: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
