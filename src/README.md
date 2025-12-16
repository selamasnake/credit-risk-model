# Source Code

This folder contains the main Python code for the credit risk model.

- **data_processing.py**: Classes and functions for data loading, exploratory data analysis (EDA), feature engineering, and preprocessing pipelines (numerical scaling, categorical encoding).  
- **proxy_target.py**: Classes and functions for creating a proxy target variable for credit risk. Includes computation of Recency, Frequency, Monetary (RFM) metrics, K-Means clustering of customers, assignment of high-risk labels, and merging the target back into the processed dataset.
