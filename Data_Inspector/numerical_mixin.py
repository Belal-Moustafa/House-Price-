import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import TOP_NUMERICAL_COLUMNS, TARGET_COL


class NumericalMixin:
    """Methods for numerical feature analysis: correlations, distributions, skewness, outliers."""

    def get_top_correlations(self, target_col: str = TARGET_COL, top_n: int = TOP_NUMERICAL_COLUMNS, plot: bool = True) -> pd.DataFrame:
        """Calculates and optionally plots the top N numerical features correlated with target."""
        numeric_df = self.numeric_df

        if target_col not in numeric_df.columns:
            raise ValueError(f"Column '{target_col}' not found in numerical columns.")

        # Calculate the correlation and re-order it
        corr = numeric_df.corr()
        top_corr_features = (corr[target_col].abs().sort_values(ascending=False).head(top_n + 1).index)
        # Getting the correlation matrix
        top_corr_matrix = corr.loc[top_corr_features, top_corr_features]

        # Visualization 
        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, cbar=True)
            plt.title(f"Correlation Matrix of Top High-Correlation Features with {target_col}", fontsize=12, pad=15)
            plt.tight_layout()
            plt.show()

    def get_high_numerical_correlations(self, threshold: float = 0.65) -> pd.DataFrame:
        """Finds pairs of numerical features that have a high correlation (Multicollinearity)."""
        corr_matrix = self.numeric_df.corr().abs()

        # تفريغ المكرر (Upper Triangle)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # تجميع الأزواج المتخطية للـ threshold
        high_corr_pairs = [
            {
                "Feature 1": col,
                "Feature 2": row,
                "Correlation": round(upper.loc[row, col], 3)
            }
            for col in upper.columns for row in upper.index
            if upper.loc[row, col] >= threshold
        ]

        if not high_corr_pairs:
            return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])

        return pd.DataFrame(high_corr_pairs).sort_values(by="Correlation", ascending=False).reset_index(drop=True)

    def plot_numerical_distribution(self, column_name: str, bins: int = 30) -> float:
        """Plots the distribution (Histogram + KDE) of a numerical column and returns its skewness."""
        numeric_df = self.numeric_df

        if column_name not in numeric_df.columns:
            raise ValueError(f"Column '{column_name}' not found in numerical columns.")

        # skewness calculations 
        skewness = float(self.df[column_name].skew())

        # Histogram with KDE
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df[column_name], kde=True, bins=bins, color="skyblue")

        plt.title(f"Distribution of {column_name} (Skewness: {skewness:.2f})", fontsize=13, pad=12)
        plt.xlabel(column_name, fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.grid(axis="y", linestyle=":", alpha=0.7)
        plt.show()

    def get_skewed_features(self, threshold: float = 0.75) -> pd.DataFrame:
        """Returns a DataFrame of numerical features with skewness exceeding the specified threshold."""
        numeric_df = self.numeric_df

        # حساب الـ Skewness لكل الأعمدة الرقمية
        skewness_series = numeric_df.skew()

        # فلترة الأعمدة التي تتجاوز القيمة المطلقة للعتبة
        skewed_features = skewness_series[skewness_series.abs() > threshold].sort_values(ascending=False)

        # تحويل النتيجة لـ DataFrame منظم
        skewed_df = pd.DataFrame({"Feature": skewed_features.index, "Skewness": skewed_features.values,}).reset_index(drop=True)

        return skewed_df

    def _get_outlier_bounds(self, column: str) -> dict:
        """Computes IQR-based mild/extreme bounds for one numerical column.

        Extracted so detect_outliers_summary and plot_outlier_scatter share the
        exact same bound calculation instead of repeating it separately.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        return {
            "IQR": IQR,
            "lower": Q1 - 1.5 * IQR,
            "upper": Q3 + 1.5 * IQR,
            "ext_lower": Q1 - 3.0 * IQR,
            "ext_upper": Q3 + 3.0 * IQR,
        }

    def detect_outliers_summary(self) -> pd.DataFrame:
        """Calculates summary of mild (1.5*IQR only), extreme (3.0*IQR), and total outliers along with boundary thresholds."""
        numeric_df = self.numeric_df
        summary_data = []

        for col in numeric_df.columns:
            bounds = self._get_outlier_bounds(col)

            if bounds["IQR"] == 0:
                continue

            lower_bound = bounds["lower"]
            upper_bound = bounds["upper"]
            ext_lower_bound = bounds["ext_lower"]
            ext_upper_bound = bounds["ext_upper"]

            # 1. الـ Extreme Outliers (خارج 3.0 * IQR)
            extreme_mask = (numeric_df[col] < ext_lower_bound) | (
                numeric_df[col] > ext_upper_bound
            )

            # 2. الـ Mild Outliers فقط (بين 1.5 و 3.0 * IQR)
            mild_mask = (
                (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
            ) & (~extreme_mask)

            mild_count = numeric_df[mild_mask].shape[0]
            extreme_count = numeric_df[extreme_mask].shape[0]
            total_outliers = mild_count + extreme_count
            total_rows = len(numeric_df)

            if total_outliers > 0:
                summary_data.append(
                    {
                        "Feature": col,
                        "Total Outliers": total_outliers,
                        "Mild Count (1.5x)": mild_count,
                        "Mild (%)": round((mild_count / total_rows) * 100, 2),
                        "Lower Bound": round(lower_bound, 2),
                        "Upper Bound": round(upper_bound, 2),
                        "Extreme Count (3.0x)": extreme_count,
                        "Extreme (%)": round(
                            (extreme_count / total_rows) * 100, 2
                        ),
                        "Ext. Lower Bound": round(ext_lower_bound, 2),
                        "Ext. Upper Bound": round(ext_upper_bound, 2),
                    }
                )

        outliers_df = pd.DataFrame(summary_data)
        if not outliers_df.empty:
            outliers_df = outliers_df.sort_values(
                by="Total Outliers", ascending=False
            ).reset_index(drop=True)

        return outliers_df

    def plot_boxplot(self, column_name: str) -> None:
        """Plots a Box Plot for a specific numerical column to visually inspect outliers."""
        numeric_df = self.numeric_df

        if column_name not in numeric_df.columns:
            raise ValueError(f"Column '{column_name}' not found in numerical columns.")

        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df[column_name], color="salmon")
        plt.title(f"Box Plot of {column_name}", fontsize=12, pad=12)
        plt.xlabel(column_name, fontsize=10)
        plt.grid(axis="x", linestyle=":", alpha=0.7)
        plt.show()

    def plot_outlier_scatter(self, feature_col: str, target_col: str = "SalePrice") -> None:
        """Plots a scatter plot of feature vs target, color-coded by outlier severity (Normal, Mild, Extreme)."""
        numeric_df = self.numeric_df

        if (feature_col not in numeric_df.columns or target_col not in numeric_df.columns):
            raise ValueError("Both columns must be numerical and present in DataFrame.")

        # حساب الـ IQR والحدود للعمود المطلوب
        bounds = self._get_outlier_bounds(feature_col)
        lower_bound = bounds["lower"]
        upper_bound = bounds["upper"]
        extreme_lower = bounds["ext_lower"]
        extreme_upper = bounds["ext_upper"]

        # تصنيف كل نقطة
        def classify_point(val):
            if val < extreme_lower or val > extreme_upper:
                return "Extreme Outlier (3.0*IQR)"
            elif val < lower_bound or val > upper_bound:
                return "Mild Outlier (1.5*IQR)"
            else:
                return "Normal"

        plot_df = self.df[[feature_col, target_col]].copy()
        plot_df["Outlier_Category"] = plot_df[feature_col].apply(classify_point)

        # الرسم البياني
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=plot_df,
            x=feature_col,
            y=target_col,
            hue="Outlier_Category",
            palette={
                "Normal": "#4C72B0",
                "Mild Outlier (1.5*IQR)": "#DD8452",
                "Extreme Outlier (3.0*IQR)": "#C44E52",
            },
            alpha=0.8,
            s=50,
        )

        plt.title(
            f"{feature_col} vs {target_col} (Outlier Severity Analysis)",
            fontsize=12,
            pad=12,
        )
        plt.xlabel(feature_col, fontsize=10)
        plt.ylabel(target_col, fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(title="Category")
        plt.show()
