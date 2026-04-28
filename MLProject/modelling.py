import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

# Buat folder untuk menyimpan model
os.makedirs("models", exist_ok=True)

rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_train_processed, y_train)

# Hitung metrics
train_score = rf.score(X_train_processed, y_train)
test_score = rf.score(X_test_processed, y_test)

# Save model dan preprocessor
model_path = "models/model.pkl"
preprocessor_path = "models/preprocessor.pkl"

joblib.dump(rf, model_path)
joblib.dump(preprocessor, preprocessor_path)

# Save metrics ke file
with open("models/metrics.txt", "w") as f:
    f.write(f"training_score: {train_score:.4f}\n")
    f.write(f"test_score: {test_score:.4f}\n")

print(f"Model berhasil dilatih!")
print(f"Training score: {train_score:.4f}")
print(f"Test score: {test_score:.4f}")
print(f"Model saved to: {model_path}")