import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def fetch_and_save_data():
    print("Fetching AI4I 2020 Predictive Maintenance dataset from UCI...")
    
    # Fetch dataset via ID 580
    ai4i_2020_predictive_maintenance = fetch_ucirepo(id=601)
    
    # Extract features and targets as pandas dataframes
    X = ai4i_2020_predictive_maintenance.data.features
    y = ai4i_2020_predictive_maintenance.data.targets
    
    # Combine them into a single comprehensive dataframe
    df = pd.concat([X, y], axis=1)
    
    # Make sure our target data/ directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to local CSV path
    output_path = os.path.join('data', 'predictive_maintenance.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Success! Dataset saved safely to: {output_path}")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    fetch_and_save_data()