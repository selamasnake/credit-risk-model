"""
train.py
Handles model training, hyperparameter tuning, and experiment tracking with MLflow.
"""

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    """Train multiple models with optional hyperparameter tuning and track experiments in MLflow."""

    def __init__(self, X, y):
        """Initialize trainer, scale features, and set up storage for models."""
        self.X = X
        self.y = y
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.best_model = None
        self.best_metrics = None
        self.models = {}  # store all trained models

    def train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and test sets with stratification."""
        return train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

    def train_and_log_models(
        self,
        models=["logistic_regression", "random_forest", "xgboost"],
        tune=True,
        random_state=42
    ):
        """Train specified models, optionally perform hyperparameter tuning, and log results to MLflow."""
        for model_name in models:
            with mlflow.start_run(run_name=model_name):
                if model_name == "logistic_regression":
                    model = LogisticRegression(random_state=random_state, max_iter=500)
                    param_grid = {"C": [0.01, 0.1, 1, 10]} if tune else {}
                elif model_name == "random_forest":
                    model = RandomForestClassifier(random_state=random_state)
                    param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10]} if tune else {}
                elif model_name == "xgboost":
                    model = XGBClassifier(
                        random_state=random_state,
                        n_estimators=100,
                        max_depth=5,
                        eval_metric="logloss"  # no use_label_encoder
                    )
                    param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5]} if tune else {}
                else:
                    raise ValueError(f"Unsupported model: {model_name}")

                # Grid search if tuning
                if tune and param_grid:
                    search = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc")
                    search.fit(self.X_scaled, self.y)
                    model = search.best_estimator_
                else:
                    model.fit(self.X_scaled, self.y)

                # Save trained model
                self.models[model_name] = model

                # Compute metrics
                preds = model.predict(self.X_scaled)
                metrics = {
                    "accuracy": accuracy_score(self.y, preds),
                    "precision": precision_score(self.y, preds),
                    "recall": recall_score(self.y, preds),
                    "f1": f1_score(self.y, preds),
                    "roc_auc": roc_auc_score(self.y, model.predict_proba(self.X_scaled)[:, 1])
                }

                # Log to MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, name="model")  # updated from artifact_path to name

                # Update best model
                if self.best_metrics is None or metrics["roc_auc"] > self.best_metrics["roc_auc"]:
                    self.best_model = model
                    self.best_metrics = metrics

    def evaluate_best_model(self, X_test, y_test):
        """Evaluate the best model on test data."""
        X_test_scaled = self.scaler.transform(X_test)
        preds = self.best_model.predict(X_test_scaled)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, self.best_model.predict_proba(X_test_scaled)[:, 1])
        }
        return self.best_model, metrics
