import pandas as pd
from IPython.display import display, Markdown

from .overview_mixin import OverviewMixin
from .missing_values_mixin import MissingValuesMixin
from .numerical_mixin import NumericalMixin
from .categorical_mixin import CategoricalMixin
from .correlation_mixin import CorrelationMixin


class DataInspector(
    OverviewMixin,
    MissingValuesMixin,
    NumericalMixin,
    CategoricalMixin,
    CorrelationMixin,
):
    """This class is responsible for performing EDA.

    All the actual analysis methods live in the Mixin classes above, grouped
    by concern (overview / missing values / numerical / categorical /
    correlation). This class just wires them together and holds the shared
    state (self.df, self.numeric_df, self.cat_df).
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("The provided DataFrame is empty.")

        # ناخد نسخة (copy) بدل ما نمسك reference للـ df الأصلي، عشان لو
        # حد غيّر في الداتا الأصلية بعد كده، الـ Inspector يفضل زي ما هو
        self.df = df.copy()

    @property
    def numeric_df(self) -> pd.DataFrame:
        """Numerical subset of self.df, recomputed on access so it always
        reflects the current self.df (e.g. after cleaning/imputation)."""
        return self.df.select_dtypes(include=["number"])

    @property
    def cat_df(self) -> pd.DataFrame:
        """Categorical/object subset of self.df, recomputed on access for
        the same reason as numeric_df above."""
        return self.df.select_dtypes(include=["object", "category"])

    def run_full_inspection(self, target_col: str = "SalePrice") -> dict:
        """Runs the most commonly-needed checks and returns them as a dict,
        so downstream steps (e.g. feature engineering) can grab everything
        in one call instead of calling each method separately.
        """
        results = {
            "missing": self.get_columns_contain_missing_values(),
            "high_missing": self.get_columns_contain_high_missing_values(),
            "skewed_features": self.get_skewed_features(),
            "outliers": self.detect_outliers_summary(),
            "feature_classification": self.classify_features(),
        }
        for name, df_result in results.items():
            display(Markdown(f"###  {name.replace('_', ' ').title()}"))
            if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                display(df_result)
            else:
                print("No data available.")
            
        return results