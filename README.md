# House Prices — Kaggle (Advanced Regression Techniques)

An end-to-end regression pipeline that predicts residential home sale
prices from the Ames, Iowa housing dataset (Kaggle's *House Prices —
Advanced Regression Techniques* competition). Built as a personal
project to practice designing an ML pipeline with proper OOP structure
rather than a single notebook.

The final model is an **XGBoost regressor**, evaluated with **Stratified
K-Fold cross-validation** and trained on a **log-transformed target**.

## Project structure

```
.
├── config.py                  # single source of truth for paths, target column, seed, hyperparameters and some other fixed numbers
├── Data_Loader.py              # loads train/test CSVs
├── Data_Inspector/             # mixin-based EDA toolkit (see "Exploratory Data Analysis" below)
│   ├── __init__.py
│   ├── overview_mixin.py
│   ├── missing_values_mixin.py
│   ├── numerical_mixin.py
│   ├── categorical_mixin.py
│   └── correlation_mixin.py
├── EDA.ipynb                   # the actual EDA session that drove the cleaning decisions below
├── housing_cleaning.py         # outlier removal, target transform, and the column-cleaning pipeline
├── train_model.py              # trains the model, cross-validates it, saves the fitted artifacts
├── predict.py                  # loads the saved artifacts, predicts on test, writes submission.csv
└── models/                     # created at runtime: final_model.joblib, cleaning_pipeline.joblib
```

## Pipeline walkthrough

### 1. Data loading — `Data_Loader.py`

A small `DataLoader` class wrapping `pd.read_csv` for the train/test
files, with a shared `_validate_path` check so a missing file fails
with a clear `FileNotFoundError` instead of a confusing pandas error
further downstream.

### 2. Exploratory Data Analysis — `Data_Inspector/` + `EDA.ipynb`

Rather than one large EDA script, the analysis logic is split into a
`DataInspector` class assembled from five mixins, each responsible for
one concern:

- `OverviewMixin` — shape, dtypes, `describe()`, duplicate rows
- `MissingValuesMixin` — missing counts/percentages, high-missing columns
- `NumericalMixin` — correlations with the target, skewness, IQR-based
  outlier detection, distribution/outlier plots
- `CategoricalMixin` — cardinality, dominance ratio, count/box plots,
  and `profile_column()`, which picks the right plot automatically and
  profiles a categorical feature against the target in one call
- `CorrelationMixin` — Cramér's V for categorical-categorical
  association, ANOVA for categorical-vs-numerical association, and
  feature type classification

Keeping this in a general-purpose class (rather than one-off EDA code)
meant it could be reused later as a *safety net* inside the cleaning
pipeline itself — see step 4.

`EDA.ipynb` is the actual notebook session where these tools
were used to inspect the data and reach the cleaning decisions below.

### 3. Outlier removal & target transform — `housing_cleaning.py`

Two operations are deliberately kept **outside** the sklearn pipeline:

- **Outlier removal** (`remove_training_outliers`) drops a handful of
  manually-identified bad rows (e.g. very large `GrLivArea` at a low
  `SalePrice`). This must only ever touch the training set — every test
  row still needs a prediction, so this can't be a `fit`/`transform`
  step.
- **Target transform** (`transform_target` / `inverse_transform_target`)
  applies `log1p` to `SalePrice` before training (`SalePrice` is
  right-skewed) and `expm1` to convert predictions back to real dollar
  values afterward.

### 4. Feature cleaning — `housing_cleaning.py`

Every column's treatment (drop / binary-encode / ordinal-map / impute /
log-transform) is defined once in a central `DECISIONS` dictionary, and
a single `DecisionBasedCleaner` (an sklearn `TransformerMixin`) applies
whichever strategy each column is assigned.

**Why the columns are named explicitly instead of decided purely from
missing %:** each strategy here was chosen empirically. `Alley`, for
example, has a perfectly reasonable sample size but was still dropped
because keeping it tested *worse* in cross-validation than removing it.
There's no way to auto-derive an empirical result from a missing-value
percentage alone, so decisions like this have to be written down
explicitly.

What *is* automatic: `DecisionBasedCleaner.fit()` re-uses
`DataInspector.get_columns_contain_high_missing_values()` (rather than
re-implementing that logic) to warn if a new high-missing column ever
shows up with no entry in `DECISIONS` — so a column can't silently slip
through unhandled if the data changes later.

A second transformer, `CategoricalTypeAligner`, converts remaining
`object` columns to pandas `category` dtype (for XGBoost's
`enable_categorical=True`), learning the category set on train and
reusing it as-is on test — a category value seen only in test becomes
`NaN`, which XGBoost simply treats as missing.

**Known gap, documented rather than silently skipped:** ideally, every
column's cleaning decision would be preceded by a dedicated,
column-specific EDA cell (a few columns, like `Fence` and
`FireplaceQu`, do have one in the notebook). In practice this wasn't
done consistently for every column — some decisions (like dropping
`PoolQC`) were made from the general missing-value overview rather than
a per-column deep-dive. This is a conscious trade-off for project scope,
not an oversight to silently paper over.

### 5. Centralized configuration — `config.py`

Paths, the target column name, the random seed, cross-validation
settings, the high-missing threshold, and the XGBoost hyperparameters
all live in one file and are imported everywhere they're needed,
instead of being repeated (and risking drifting out of sync) across
`train_model.py`, `predict.py`, and `housing_cleaning.py`.

The per-column `DECISIONS` table is **not** in `config.py` on purpose —
those are column-specific empirical judgment calls (see step 4), not
general project settings.

### 6. Model training — `train_model.py`

1. Load train data, remove outlier rows, log-transform the target
2. Fit the cleaning pipeline on the training features only
3. Run **Stratified K-Fold cross-validation**, with folds stratified on
   `SalePrice` binned into quantiles (`pd.qcut`) — this keeps the price
   distribution consistent across folds, which a plain K-Fold split
   doesn't guarantee for a continuous target. This CV loop exists purely
   to get a trustworthy performance estimate; none of the fold models
   are kept.
4. Refit **one final model** on the entire training set — this is the
   model that actually gets saved and used for predictions
5. Save the fitted `final_model` and the fitted `cleaning_pipeline` to
   `models/` with `joblib`

### 7. Inference & submission — `predict.py`

Loads the two saved artifacts and:

1. Reads the test CSV and **saves the `Id` column aside before
   transforming** — the cleaning pipeline drops `Id`, but the
   submission file needs it back
2. Calls `cleaning_pipeline.transform()` (never `fit_transform`) so the
   test set gets *exactly* the same treatment and category sets learned
   on train
3. Predicts (still on the log1p scale) and applies
   `inverse_transform_target` to get real dollar predictions
4. Writes `submission.csv` with exactly the columns Kaggle expects for
   this competition — `Id`, `SalePrice`, no index column


### Model evaluation:
- Train CV MAE: $14,822.95
- Test MAE: $14,910.377         (after submitting the test submission and getting the score from kaggle)



## How to run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy joblib ipython

# 1. Edit config.py: set TRAIN_PATH and TEST_PATH to your local CSV locations
# 2. Train + save the model and cleaning pipeline
python train_model.py

# 3. Generate submission.csv from the saved artifacts
python predict.py
```

## Known limitations / possible next steps

- **No feature engineering yet** — every cleaning step transforms an
  *existing* column; nothing combines columns into new ones (e.g. total
  square footage, house age at sale).
- **No automated tests** — correctness has so far been verified by
  running the pipeline end-to-end and inspecting outputs manually.
- **Column-specific EDA before cleaning decisions** wasn't done for
  every single column (see step 4 above).
