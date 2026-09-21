import pandas as pd


class OverviewMixin:
    """High-level dataset summary methods: shape, dtypes, describe, duplicates."""

    def get_summary(self) -> None:
        """Prints a high-level summary of the dataset."""
        print("=== Dataset Dimensions ===")
        print(f"Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}\n")

        print("=== Data Types Breakdown ===")
        print(self.df.dtypes.value_counts())

        print("\n=== Detailed Information ===")
        self.df.info()

    def describe_numerical(self) -> pd.DataFrame:
        """Returns summary statistics for numerical columns."""
        return self.df.describe()

    def describe_categorical(self) -> pd.DataFrame:
        """Returns summary statistics for categorical/object columns."""
        return self.df.describe(include=["object", "category"])

    def check_duplicates(self, subset_cols=None):
        """Checks for duplicate rows in the dataset or based on specific columns."""
        duplicate_count = self.df.duplicated(subset=subset_cols).sum()
        total_rows = len(self.df)
        duplicate_pct = (duplicate_count / total_rows) * 100

        print(f"Total Duplicate Rows: {duplicate_count} ({duplicate_pct:.2f}%)")

        if duplicate_count > 0:
            print("Found duplicates! Consider removing or inspecting them.")
        else:
            print("No duplicate rows found.")

        return duplicate_count
