import os
import pandas as pd
from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE


class DataLoader:
    """Load and save CSV datasets."""

    def __init__(self, path: str | None = None):
        """Initialize with optional default file path."""
        self.path = path
        self.df = None

    def load_data(self, path: str | None = None) -> pd.DataFrame:
        """Load CSV into DataFrame. Returns the DataFrame."""
        file_path = path or self.path
        if not file_path:
            raise ValueError("No file path specified.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.df = pd.read_csv(file_path, low_memory=False)
        return self.df

    def save_data(self, output_path: str) -> None:
        """Save current DataFrame to CSV."""
        if self.df is None:
            raise ValueError("No data loaded.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)


class EDAProcessor:
    """Compute EDA metrics for numerical and categorical features."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data

    def data_overview(self) -> dict:
        """Returns dataset information and dtypes."""

        return self.data.info()

    def missing_values(self) -> pd.DataFrame:
        """Checks for missing counts and percentages per column."""

        missing = self.data.isna().sum()
        return pd.DataFrame({
            "missing_count": missing,
            "missing_pct": missing / len(self.data)
        })

    def summary_statistics(self, numerical_cols: list[str]) -> pd.DataFrame:
        """Calculates summary stats + skewness/kurtosis for numerical features."""

        desc = self.data[numerical_cols].describe().T
        desc["skew"] = self.data[numerical_cols].skew()
        desc["kurtosis"] = self.data[numerical_cols].kurtosis()
        return desc


    def outlier_metrics(self, numerical_cols: list[str]) -> pd.DataFrame:
        """Calculate IQR-based outlier bounds and counts."""

        results = {}
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((self.data[col] < lower) | (self.data[col] > upper)).sum()
            results[col] = {
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": outliers,
                "outlier_pct": outliers / len(self.data)
            }
        return pd.DataFrame(results).T

    def categorical_metrics(self, categorical_cols: list[str]) -> pd.DataFrame:
        """Returns number of categories, top category share, and entropy."""

        summaries = {}
        for col in categorical_cols:
            counts = self.data[col].value_counts()
            probs = counts / counts.sum()
            summaries[col] = {
                "num_categories": counts.shape[0],
                "top_category_pct": probs.iloc[0],
                "entropy": entropy(probs)
            }
        return pd.DataFrame(summaries).T

    def correlation_matrix(self, numerical_cols: list[str], method: str = "pearson") -> pd.DataFrame:
        """Returns correlation matrix for numerical features."""

        return self.data[numerical_cols].corr(method=method)


class FeatureEngineer:
    """Feature engineering for transaction-level credit data."""

    NUMERICAL_FEATURES = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    ]

    CATEGORICAL_FEATURES = [
        "CurrencyCode",
        "CountryCode",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]

    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"], errors="coerce"
        )

        df["transaction_hour"] = df["TransactionStartTime"].dt.hour
        df["transaction_day"] = df["TransactionStartTime"].dt.day
        df["transaction_month"] = df["TransactionStartTime"].dt.month
        df["transaction_year"] = df["TransactionStartTime"].dt.year

        return df

    def aggregate_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        agg_df = (
            df.groupby("CustomerId")
            .agg(
                total_transaction_amount=("Amount", "sum"),
                avg_transaction_amount=("Amount", "mean"),
                transaction_count=("TransactionId", "count"),
                std_transaction_amount=("Amount", "std"),
            )
            .reset_index()
        )

        # Handle single-transaction customers
        agg_df["std_transaction_amount"] = agg_df["std_transaction_amount"].fillna(0)

        return agg_df

    @staticmethod
    def build_preprocessing_pipeline() -> ColumnTransformer:
        """Create sklearn preprocessing pipeline."""

        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, FeatureEngineer.NUMERICAL_FEATURES),
                ("cat", categorical_pipeline, FeatureEngineer.CATEGORICAL_FEATURES),
            ]
        )

        return preprocessor


class WoETransformer:
    """
    Weight of Evidence (WoE) transformer for binary classification tasks.
    It wraps the xverse.WOE class to handle multiple features.
    """
    def __init__(self, target_col: str, features: list = None):
        self.target_col = target_col
        self.features = features
        self.woe_map = {}
        self.feature_names_ = None

    def fit(self, df: pd.DataFrame):
        y = df[self.target_col]
        
        if y.nunique() != 2:
            raise ValueError("WoE requires a binary target variable.")

        if self.features is None:
            X_df = df.select_dtypes(include=["number"]).drop(
                columns=[self.target_col], errors="ignore"
            )
        else:
            X_df = df[self.features]

        self.feature_names_ = X_df.columns.tolist()

        for feature in self.feature_names_:
            feature_woe = WOE()
            X_feature = X_df[[feature]]
            feature_woe.fit(X_feature, y)
            self.woe_map[feature] = feature_woe
            
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("WoETransformer has not been fitted yet.")
            
        transformed_dfs = []
        
        for feature in self.feature_names_:
            X_feature = df[[feature]]
            feature_woe = self.woe_map[feature]
            transformed_series = feature_woe.transform(X_feature)
            transformed_df = pd.DataFrame(transformed_series, columns=[feature])
            transformed_dfs.append(transformed_df)
            
        result_df = pd.concat(transformed_dfs, axis=1)
        result_df.index = df.index
        
        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
