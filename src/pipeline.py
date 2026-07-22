import os
import pandas as pd
import mlflow
import mlflow.sklearn
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from prefect import task, flow

@task(name="Fetch UCI Data", retries=2, retry_delay_seconds=5)
def fetch_data(data_path: str) -> pd.DataFrame:
    """Task 1: Fetch raw dataset from UCI repo and save locally."""
    print("Fetching dataset from UCI repository...")
    ai4i_2020 = fetch_ucirepo(id=580)
    X = ai4i_2020.data.features
    y = ai4i_2020.data.targets
    
    df = pd.concat([X, y], axis=1)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Data saved to {data_path} (Shape: {df.shape})")
    return df

@task(name="Preprocess Data")
def preprocess_data(df: pd.DataFrame):
    """Task 2: Clean column names and split features/target."""
    # Clean column names
    df.columns = (
        df.columns.str.lower()
        .str.replace(' ', '_')
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
    )
    
    # Drop irrelevant non-numeric / auxiliary columns
    drop_cols = ['udi', 'product_id', 'type', 'twf', 'hdf', 'pwf', 'osf', 'rnf']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df_clean = df.drop(columns=existing_drop_cols)
    
    X = df_clean.drop(columns=['machine_failure'])
    y = df_clean['machine_failure']
    
    return X, y

@task(name="Train & Evaluate Model")
def train_model(X: pd.DataFrame, y: pd.Series):
    """Task 3: Train Random Forest and log results to MLflow."""
    mlflow.set_experiment("Predictive_Maintenance_Orchestrated")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced"
    }
    
    with mlflow.start_run(run_name="Prefect_Orchestrated_RF"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred))
        }
        
        # Log params, metrics, and model artifact to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            input_example=X_train.head(1)
        )
        
        print(f"Model successfully logged to MLflow! Metrics: {metrics}")
        return metrics

@flow(name="Predictive Maintenance End-to-End Pipeline")
def main_flow():
    """Main orchestrated flow chaining tasks together."""
    data_path = os.path.join('data', 'predictive_maintenance.csv')
    
    # Dependency chain: Data Ingestion -> Preprocessing -> Training
    df = fetch_data(data_path)
    X, y = preprocess_data(df)
    metrics = train_model(X, y)
    print("Pipeline Execution Complete!")

if __name__ == "__main__":
    main_flow()