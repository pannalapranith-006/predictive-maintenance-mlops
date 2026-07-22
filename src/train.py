import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score

def load_and_preprocess_data(data_path):
    """Loads data, cleans column names, and splits into features and target."""
    df = pd.read_csv(data_path)
    
    # Clean column names (lowercase, replace spaces/brackets)
    df.columns = (
        df.columns.str.lower()
        .str.replace(' ', '_')
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
    )
    
    # Drop identifier/text columns that aren't numeric sensor values
    drop_cols = ['udi', 'product_id', 'type', 'twf', 'hdf', 'pwf', 'osf', 'rnf']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df_clean = df.drop(columns=existing_drop_cols)
    
    # Split target (machine_failure) from feature columns
    X = df_clean.drop(columns=['machine_failure'])
    y = df_clean['machine_failure']
    
    return X, y

def run_training():
    # 1. Set up MLflow Experiment Name
    mlflow.set_experiment("Predictive_Maintenance_Baseline")
    
    data_path = os.path.join('data', 'predictive_maintenance.csv')
    X, y = load_and_preprocess_data(data_path)
    
    # 2. Train/Test Split (Stratify ensures rare 3.4% failure class is preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced"  # Crucial for handling class imbalance!
    }
    
    # 3. Start an MLflow Run Context
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        print("Training Random Forest model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict probabilities and hard classes
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"--- Model Metrics ---")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"F1 Score:      {f1:.4f}")
        print(f"Precision:     {precision:.4f}")
        print(f"Recall:        {recall:.4f}")
        
        # 4. Log Parameters and Metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        })
        
        # 5. Log Model Artifact to MLflow
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            input_example=X_train.head(1)
        )
        
        print("Successfully logged model, parameters, and metrics to MLflow!")

if __name__ == "__main__":
    run_training()