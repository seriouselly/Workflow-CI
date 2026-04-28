import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Mengambil kredensial dari GitHub Secrets
dagshub_user = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

# Set MLflow tracking URI dan credentials untuk DagShub
if dagshub_user and dagshub_token:
    mlflow_uri = f"https://dagshub.com/{dagshub_user}/Workflow-CI-V2/mlflow"
    mlflow.set_tracking_uri(mlflow_uri)
    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# Set experiment
mlflow.set_experiment("students-performance-prediction")

df = pd.read_csv('../dataset_raw/StudentsPerformance.csv')
X = df.drop(columns=['math score'])
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['reading score', 'writing score']
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

with mlflow.start_run():
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train_processed, y_train)
    
    # Mencatat model agar nanti bisa dibungkus oleh Docker
    mlflow.sklearn.log_model(rf, "model")
    print("Model berhasil dilatih dan dicatat oleh Robot!")