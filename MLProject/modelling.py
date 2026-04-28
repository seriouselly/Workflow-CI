import pandas as pd
import os
import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set MLflow tracking URI ke folder mlruns di current directory
current_dir = Path(__file__).parent
mlflow_dir = current_dir / "mlruns"
mlflow_dir.mkdir(exist_ok=True)
mlflow.set_tracking_uri(f"file:{mlflow_dir.absolute()}")

# MLflow akan menggunakan local tracking (mlruns/)
# Credentials dari GitHub Secrets bisa digunakan untuk integrasi lanjutan jika diperlukan
dagshub_user = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

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