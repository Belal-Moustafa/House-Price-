from config import HIGH_MISSING_THRESHOLD
import pandas as pd


class MissingValuesMixin:
    """Methods for detecting and summarizing missing values."""

    def get_columns_contain_missing_values(self) -> pd.DataFrame:
        """Returns a DataFrame with columns containing missing values, their counts, and percentages, sorted in descending order."""
        total_missing = self.df.isnull().sum()
        percent_missing = (total_missing / len(self.df)) * 100

        missing_df = (
            pd.DataFrame({
                "Missing Count": total_missing,
                "Percentage (%)": percent_missing
                }).query("`Missing Count` > 0").sort_values(by="Percentage (%)", ascending=False))

        return missing_df

    def get_columns_contain_high_missing_values(self, threshold_percent: float = HIGH_MISSING_THRESHOLD) -> pd.DataFrame:
        """Returns a DataFrame of columns with missing values exceeding the specified threshold."""
        # Calling the first method and then we will extract the high missing values columns only
        missing_df = self.get_columns_contain_missing_values()

        # Choosing high missing values columns from the general method "get_columns_contain_missing_values"
        high_missing_df = missing_df.query("`Percentage (%)` >= @threshold_percent")

        return high_missing_df
