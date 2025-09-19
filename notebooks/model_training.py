# model_training.py

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# === Step 1: Load Preprocessed Data ===
print("🔹 Loading data...")
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')  # we'll use this only for evaluation

print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# === Step 2: Build Isolation Forest Model ===
print("\n🔹 Training Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_train)
print("✅ Isolation Forest model trained!")

# === Step 3: Predict Anomalies ===
# -1 = anomaly, 1 = normal
y_pred_test = iso_forest.predict(X_test)

# Map: -1 → 1 (Attack), 1 → 0 (Normal)
y_pred_test = np.where(y_pred_test == -1, 1, 0)

# For ground truth: Map actual labels (0 = Normal, 1 = Attack)
y_true = np.where(y_test == 5, 0, 1)  # Adjust this based on your label mapping

# === Step 4: Evaluation ===
print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_test))

print("\n🔹 Classification Report:")
print(classification_report(y_true, y_pred_test, target_names=['Normal', 'Attack']))
