import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ProxyTargetEngineer:
    """
    Compute a proxy target variable for credit risk based on RFM clustering.
    """

    def __init__(self, snapshot_date: pd.Timestamp = None, n_clusters: int = 3, random_state: int = 42):
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.high_risk_cluster_ = None

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Recency, Frequency, and Monetary metrics per customer.
        """
        if self.snapshot_date is None:
            self.snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

        rfm = (
            df.groupby("CustomerId")
            .agg(
                recency=("TransactionStartTime", lambda x: (self.snapshot_date - x.max()).days),
                frequency=("TransactionId", "count"),
                monetary=("Amount", "sum")
            )
            .reset_index()
        )
        return rfm

    def cluster_customers(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Scale RFM features and perform KMeans clustering.
        """
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        rfm["cluster"] = kmeans.fit_predict(rfm_scaled)
        return rfm

    def assign_high_risk(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Identify the high-risk cluster and create a binary target column 'is_high_risk'.
        """
        # Business logic: low frequency & low monetary = high risk
        cluster_stats = rfm.groupby("cluster")[["frequency", "monetary"]].mean()
        self.high_risk_cluster_ = cluster_stats.mean(axis=1).idxmin()

        rfm["is_high_risk"] = (rfm["cluster"] == self.high_risk_cluster_).astype(int)
        return rfm

    def merge_target(self, processed_df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the binary target column 'is_high_risk' into the processed dataset.
        """
        merged_df = processed_df.merge(
            rfm[["CustomerId", "is_high_risk"]],
            on="CustomerId",
            how="left"
        )
        return merged_df

    def fit_transform(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: compute RFM, cluster, assign high-risk, and merge into processed data.
        """
        rfm = self.compute_rfm(raw_df)
        rfm = self.cluster_customers(rfm)
        rfm = self.assign_high_risk(rfm)
        final_df = self.merge_target(processed_df, rfm)
        return final_df
