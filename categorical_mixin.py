import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CategoricalMixin:
    """Methods for categorical feature analysis: cardinality, dominance, distributions."""

    def get_categorical_summary(self) -> pd.DataFrame:
        """Returns a detailed summary DataFrame for categorical columns (Cardinality, Top Class, Dominance Ratio)."""
        cat_df = self.cat_df

        if cat_df.empty:
            print("No categorical columns found.")
            return pd.DataFrame()

        summary_data = []
        total_rows = len(self.df)

        # إعادة استخدام النتيجة الجاهزة من MissingValuesMixin بدل حساب الـ missing من جديد
        missing_lookup = self.get_columns_contain_missing_values()["Missing Count"]

        for col in cat_df.columns:
            cardinality = cat_df[col].nunique(dropna=True)
            missing_count = missing_lookup.get(col, 0)
            missing_pct = round((missing_count / total_rows) * 100, 2)

            top_value = cat_df[col].mode()[0] if not cat_df[col].mode().empty else None
            top_freq = cat_df[col].value_counts().max() if cardinality > 0 else 0
            top_ratio = round((top_freq / total_rows) * 100, 2)

            summary_data.append({
                "Feature": col,
                "Unique Classes": cardinality,
                "Missing Count": missing_count,
                "Missing (%)": missing_pct,
                "Top Class": top_value,
                "Top Frequency": top_freq,
                "Top Dominance (%)": top_ratio
            })

        summary_df = pd.DataFrame(summary_data).sort_values(by="Unique Classes", ascending=False).reset_index(drop=True)
        return summary_df

    def plot_categorical_count(self, column_name: str, top_n: int = 15) -> None:
        """Plots the count distribution of categories in a column (limited to top_n for readability)."""
        cat_df = self.cat_df

        if column_name not in cat_df.columns:
            raise ValueError(f"Column '{column_name}' not found in categorical columns.")

        plt.figure(figsize=(10, 5))
        # 1. تجهيز السلسلة وتحويل الـ NaN لنص واضح للرسم
        series_to_plot = self.df[column_name].fillna("Missing (NaN)")

        # 2. تحديد الترتيب بناءً على التكرار متضمناً الـ NaN
        order = series_to_plot.value_counts().head(top_n).index
        sns.countplot(data=self.df, y=series_to_plot, order=order, palette="viridis", hue=series_to_plot, legend=False)

        plt.title(f"Top {top_n} Categories in '{column_name}'", fontsize=12, pad=12)
        plt.xlabel("Count", fontsize=10)
        plt.ylabel(column_name, fontsize=10)
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.show()

    #this function is to be used for columns which contain high missing values because if we used the previous one the nan will be dominant
    def plot_categorical_high_missing_values_count(self, column_name: str, top_n: int = 15) -> None:
        """Plots the count distribution of categories in a column (limited to top_n for readability)."""
        cat_df = self.cat_df

        if column_name not in cat_df.columns:
            raise ValueError(f"Column '{column_name}' not found in categorical columns.")

        plt.figure(figsize=(10, 5))

        order = self.df[column_name].value_counts().head(top_n).index
        sns.countplot(data=self.df, y=column_name, order=order, palette="viridis")

        plt.title(f"Top {top_n} Categories in '{column_name}'", fontsize=12, pad=12)
        plt.xlabel("Count", fontsize=10)
        plt.ylabel(column_name, fontsize=10)
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.show()

    def profile_column(self, column_name: str, target_col: str = "SalePrice", high_missing_threshold: float = 40.0) -> None:
        """One call instead of the usual 2-3: picks the right count-plot
        automatically (the NaN-dominant variant if this column's missing %
        is above the threshold, the normal one otherwise) using
        MissingValuesMixin's own numbers, then plots it against the target.

        Doesn't duplicate any plotting logic — it just decides which
        existing method to call.
        """
        missing_df = self.get_columns_contain_missing_values()
        missing_pct = missing_df["Percentage (%)"].get(column_name, 0.0)

        if missing_pct >= high_missing_threshold:
            self.plot_categorical_high_missing_values_count(column_name)
        else:
            self.plot_categorical_count(column_name)

        self.plot_categorical_vs_target(column_name, target_col=target_col)

    def plot_categorical_vs_target(self, feature_col: str, target_col: str = "SalePrice", top_n: int = 10) -> None:
        """Plots a BoxPlot layered with Stripplot to reveal rare categories and sample counts."""
        top_categories = self.df[feature_col].value_counts().head(top_n).index
        filtered_df = self.df[self.df[feature_col].isin(top_categories)]
        sorted_order = filtered_df.groupby(feature_col)[target_col].median().sort_values(ascending=False).index

        plt.figure(figsize=(12, 6))

        # رسم الـ Boxplot الأساسي
        sns.boxplot(data=filtered_df, x=feature_col, y=target_col, order=sorted_order, palette="Set2", showfliers=False)

        # إضافة النقط الحقيقية فوق الصناديق لتوضيح الفئات القليلة
        sns.stripplot(data=filtered_df, x=feature_col, y=target_col, order=sorted_order, color="black", alpha=0.3, jitter=0.2)

        plt.title(f"{target_col} Distribution by '{feature_col}' (Points show sample density)", fontsize=12, pad=12)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def get_unique_categories_map(self, column_name: str = None, strip_spaces: bool = True) -> pd.DataFrame:
        """
        Returns all unique text values per categorical column without truncation (...).
        Optionally strips leading/trailing spaces to inspect true unique entries.
        """
        cat_df = self.cat_df

        if cat_df.empty:
            print("No categorical columns found.")
            return pd.DataFrame()

        # ضبط إعدادات العرض لبانداز عشان ما يقطعش الكلام بـ (...)
        pd.set_option('display.max_colwidth', None)

        def clean_and_get_uniques(series):
            # تحويل القيم لنصوص وتنظيف المسافات الزائدة
            if strip_spaces and series.dtype == "object":
                cleaned_series = series.dropna().astype(str).str.strip()
            else:
                cleaned_series = series.dropna()

            # استخراج القيم الفرعية وترتيبها أبتدائياً
            return sorted(cleaned_series.unique().tolist())

        # لو اخترنا عمود واحد فقط
        if column_name:
            if column_name not in cat_df.columns:
                raise ValueError(f"Column '{column_name}' is not in categorical columns.")

            uniques = clean_and_get_uniques(self.df[column_name])
            return pd.DataFrame({
                "Feature": [column_name],
                "Unique Count": [len(uniques)],
                "Values": [uniques]
            })

        # لو على كل الأعمدة
        summary = []
        for col in cat_df.columns:
            uniques = clean_and_get_uniques(self.df[col])
            summary.append({
                "Feature": col,
                "Unique Count": len(uniques),
                "Values": uniques
            })

        return pd.DataFrame(summary)
