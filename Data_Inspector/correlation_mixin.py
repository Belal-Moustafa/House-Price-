import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class CorrelationMixin:
    """Methods for categorical-categorical and categorical-numerical association, and feature classification."""

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculates Cramér's V statistic for categorical-categorical association."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape

        # تصحيح الانحياز (Bias correction)
        phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
        rcorr = r - ((r-1)**2) / (n-1)
        kcorr = k - ((k-1)**2) / (n-1)

        if min((kcorr-1), (rcorr-1)) == 0:
            return 0.0
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    def get_categorical_correlation(self, threshold: float = 0.5, top_n: int = 15, plot: bool = True) -> pd.DataFrame:
        """Computes Cramér's V matrix for all categorical features and highlights strong associations."""
        cat_df = self.cat_df

        if cat_df.shape[1] < 2:
            print("Not enough categorical columns to compute associations.")
            return pd.DataFrame()

        # اختيار أول N عمود فقط لتسهيل القراءة البصرية
        selected_cols = cat_df.columns[:top_n]
        sub_df = cat_df[selected_cols]

        cols = sub_df.columns
        cramer_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

        for col1 in cols:
            for col2 in cols:
                if col1 == col2:
                    cramer_matrix.loc[col1, col2] = 1.0
                else:
                    cramer_matrix.loc[col1, col2] = self._cramers_v(sub_df[col1].fillna("Missing"), sub_df[col2].fillna("Missing"))

        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cramer_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=True)
            plt.title(f"Categorical Association Matrix (Cramér's V) - Top {top_n} Features", fontsize=12, pad=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

        # استخراج العلاقات القوية عبر الداتاست كلها (مش بس الـ top_n)
        all_cols = cat_df.columns
        strong_pairs = []
        for i in range(len(all_cols)):
            for j in range(i + 1, len(all_cols)):
                val = self._cramers_v(cat_df[all_cols[i]].fillna("Missing"), cat_df[all_cols[j]].fillna("Missing"))
                if val >= threshold:
                    strong_pairs.append({
                        "Feature 1": all_cols[i],
                        "Feature 2": all_cols[j],
                        "Cramer's V": round(val, 3)
                    })

        if not strong_pairs:
            return pd.DataFrame(columns=["Feature 1", "Feature 2", "Cramer's V"])

        return pd.DataFrame(strong_pairs).sort_values(by="Cramer's V", ascending=False).reset_index(drop=True)

    def get_categorical_vs_numerical_correlation(self, target_col: str = "SalePrice") -> pd.DataFrame:
        """Computes ANOVA F-statistic and p-values for all categorical features against a continuous numerical target."""
        cat_df = self.cat_df

        if cat_df.empty or target_col not in self.df.columns:
            print("Categorical columns or target column not found.")
            return pd.DataFrame()

        results = []
        clean_df = self.df.dropna(subset=[target_col])

        for col in cat_df.columns:
            # تجميع قيم التارجت لكل فئة بعد استبعاد الـ NaN
            groups = [group[target_col].values for _, group in clean_df.groupby(col) if len(group) > 1]

            if len(groups) > 1:
                # إجراء اختبار ANOVA
                f_stat, p_val = stats.f_oneway(*groups)
                results.append({
                    "Feature": col,
                    "F-Statistic": round(f_stat, 2),
                    "p-value": p_val,
                    "Is Significant": p_val < 0.05
                })

        # ترتيب الأعمدة حسب قوة التأثير الإحصائي (F-Statistic)
        result_df = pd.DataFrame(results).sort_values(by="F-Statistic", ascending=False).reset_index(drop=True)
        return result_df

    def classify_features(self, ordinal_cols: list = None, discrete_threshold: int = 15) -> pd.DataFrame:
        """
        Classifies features into Qualitative (Nominal, Ordinal)
        and Quantitative (Discrete, Continuous).
        """
        ordinal_cols = ordinal_cols or []
        classification = []

        for col in self.df.columns:
            dtype = self.df[col].dtype
            n_unique = self.df[col].nunique()

            # 1. Qualitative (Categorical)
            if dtype == 'object' or dtype.name == 'category':
                if col in ordinal_cols:
                    feat_type = "Qualitative - Ordinal"
                else:
                    feat_type = "Qualitative - Nominal"

            # 2. Quantitative (Numerical)
            elif np.issubdtype(dtype, np.number):
                if col in ordinal_cols:
                    feat_type = "Qualitative - Ordinal (Encoded)"
                elif n_unique <= discrete_threshold:
                    feat_type = "Quantitative - Discrete"
                else:
                    feat_type = "Quantitative - Continuous"
            else:
                feat_type = "Other"

            classification.append({
                "Feature": col,
                "DataType": str(dtype),
                "Unique Values": n_unique,
                "Classification": feat_type
            })

        return pd.DataFrame(classification)
