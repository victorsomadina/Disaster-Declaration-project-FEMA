import mlflow
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.compose import ColumnTransformer

load_dotenv()

# tracking uri
# experiment name

MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()

EXPERIMENT_NAME = os.getenv("EXPERIMENT", "fallback_cost_prediction")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "processed_disasters.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Numerical_data: {df.select_dtypes(include=['int64', 'float64']).columns}")
    print(f"Categorical_data: {df.select_dtypes(include=['O']).columns}")
    return df


CATEGORICAL_FEATURES = ['declarationType', 
                        'incidentType',
                        'designatedArea', 'region']

NUMERICAL_FEATURES = [
     'fyDeclared', 
     'incident_duration_days', 'declaration_year', 'declaration_month',
       'is_high_cost_type', 'rolling_5yr_freq',
       'federal_share_obligated', 'project_count', 'avg_project_amount',
]

DATE_FEATURES = [
    'declarationDate',
       'incidentBeginDate', 'incidentEndDate'
]

TARGET = "total_obligated_amount"

class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        result = []
        for col in X.columns:
            dt = pd.to_datetime(X[col], errors="coerce")
            result.append(pd.DataFrame({
                f"{col}_year": dt.dt.year,
                f"{col}_month": dt.dt.month,
                f"{col}_day": dt.dt.day,
                f"{col}_dayofweek": dt.dt.dayofweek,
            }))
        return pd.concat(result, axis=1).values


def preprocess_date(DATE_FEATURES):
    return DateFeatureTransformer()

def preprocessor():
    return ColumnTransformer(
        transformers= [
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES), 
            ("date", preprocess_date(DATE_FEATURES=DATE_FEATURES), DATE_FEATURES)
        ]
    )


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"rmse": rmse, "mae": mae, "r2":r2}

def train_and_log(
        name: str,
        estimator,
        X_train,
        X_test,
        y_train,
        y_test,
        params: dict
):
    prep = preprocessor()
    pipeline = Pipeline([("prep", prep), ("estimator", estimator)])

    print('preprocessed')

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
    return pipeline, metrics


def run_train(test_size: float = 0.2, random_state: int = 42):
    df = load_data()

    cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DATE_FEATURES + [TARGET]
    df = df.dropna(subset=[c for c in cols if c in df.columns])

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DATE_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    pipeline, metrics = train_and_log(
        name="LinearRegression_run",
        estimator=LinearRegression(),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params={"random_state": random_state}
    )

    return {"model": "Linear Regression", "metrics": metrics}

if __name__ == "__main__":
    summary = run_train()
    print(summary)
