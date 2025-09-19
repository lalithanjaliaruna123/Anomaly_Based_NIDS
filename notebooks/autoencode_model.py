# autoencoder_model.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

# === Step 1: Load Preprocessed Data ===
print("🔹 Loading data...")
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')  # Only used for evaluation

print(f"✅ Data loaded: Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# === Step 2: Build Autoencoder Model ===
input_dim = X_train.shape[1]

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(input_dim, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

print("\n🔹 Model Summary:")
model.summary()

# === Step 3: Train Autoencoder on Normal Traffic Only ===
# First, filter out Normal Traffic from training set (label = 5 for Normal)
normal_indices = np.where(y_test == 5)[0]
X_normal_test = X_test[normal_indices]

print(f"\n✅ Training on Normal Traffic only: {X_train.shape}")

history = model.fit(X_train, X_train, 
                    epochs=10, 
                    batch_size=512, 
                    validation_data=(X_normal_test, X_normal_test),
                    verbose=1)

# === Step 4: Calculate Reconstruction Error ===
X_test_pred = model.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Plot Reconstruction Error Distribution
plt.figure(figsize=(8,5))
sns.histplot(mse, bins=50, kde=True, color='purple')
plt.title('Reconstruction Error Distribution')
plt.xlabel('MSE (Reconstruction Error)')
plt.tight_layout()
plt.savefig('notebooks/autoencoder_error_distribution.png')
plt.show()

# === Step 5: Set Threshold Manually (You can tune this) ===
threshold = np.percentile(mse, 95)  # Top 5% errors = anomalies
print(f"\n🔹 Threshold (95th percentile): {threshold:.6f}")

# Predict Anomalies: 1 = Attack, 0 = Normal
y_pred_auto = (mse > threshold).astype(int)

# Map ground truth: 0 = Normal, 1 = Attack
y_true = np.where(y_test == 5, 0, 1)

# === Step 6: Evaluate Model ===
print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_auto))

print("\n🔹 Classification Report:")
print(classification_report(y_true, y_pred_auto, target_names=['Normal', 'Attack']))

# === Step 7: Save Model (Optional) ===
model.save('models/autoencoder_nids.h5')
print("\n✅ Autoencoder model saved!")
