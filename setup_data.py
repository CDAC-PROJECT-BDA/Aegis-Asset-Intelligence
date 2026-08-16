import os
import zipfile
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def setup_data():
    print("Setting up dataset and training model...")
    data_dir = 'data'
    zip_path = os.path.join(data_dir, 'pump-sensor-data.zip')
    csv_path = os.path.join(data_dir, 'sensor.csv')

    if not os.path.exists(csv_path):
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        else:
            print("ERROR: Kaggle dataset not found. Please ensure sensor.csv is in the data/ folder.")
            return

    print("Loading dataset (this may take a moment for 220k rows)...")
    df = pd.read_csv(csv_path)


    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    cols_to_drop = ['timestamp', 'sensor_15']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])


    status_map = {'NORMAL': 0, 'RECOVERING': 1, 'BROKEN': 2}
    if df['machine_status'].dtype == object:
        df['machine_status'] = df['machine_status'].map(status_map)

    df = df.dropna(subset=['machine_status'])


    
    X = df.drop('machine_status', axis=1)
    y = df['machine_status']

    print("Training XGBoost Classifier...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    pipeline.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    model_path = os.path.join('src', 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    setup_data()
