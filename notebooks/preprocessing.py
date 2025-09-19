# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# === STEP 1: Load the dataset ===
print("🔹 Loading dataset...")
df = pd.read_csv("data/CICIDS2017_cleaned.csv")
print("✅ Dataset loaded successfully!")
print(f"Data shape: {df.shape}")

# === STEP 2: Label Encode the 'Attack Type' column ===
print("\n🔹 Encoding attack labels...")
label_encoder = LabelEncoder()
df['Attack_Type_Label'] = label_encoder.fit_transform(df['Attack Type'])

# Show mapping (e.g., 0 = Bot, 1 = DDoS, etc.)
print("\nAttack Type Label Mapping:")
for idx, label in enumerate(label_encoder.classes_):
    print(f"{idx} = {label}")

# === STEP 3: Drop irrelevant or risky columns ===
print("\n🔹 Dropping unnecessary columns...")
columns_to_drop = ['Attack Type', 'Destination Port']
df.drop(columns=columns_to_drop, inplace=True)
print(f"Remaining columns: {len(df.columns)}")

# === STEP 4: Separate features (X) and labels (y) ===
print("\n🔹 Splitting features and labels...")
X = df.drop(columns=['Attack_Type_Label'])
y = df['Attack_Type_Label']
print(f"✅ X shape: {X.shape}, y shape: {y.shape}")

# === STEP 5: Normalize the features ===
print("\n🔹 Scaling features with MinMaxScaler...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled to range 0–1")

# === STEP 6: Train-test split ===
print("\n🔹 Splitting data into training and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✅ Training Set: {X_train.shape}, Testing Set: {X_test.shape}")

# === STEP 7: Save preprocessed data (optional but recommended) ===
print("\n🔹 Saving preprocessed data to files...")
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)
print("✅ Preprocessed data saved to 'data/' folder!")

print("\n🎉 Preprocessing completed successfully!")
