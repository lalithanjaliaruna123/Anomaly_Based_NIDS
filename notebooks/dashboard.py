import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# ────────────────────────────────────────────────
# Streamlit UI Setup
st.set_page_config(page_title="NIDS Anomaly Detection", layout="wide")
st.title("🔍 Anomaly Detection Dashboard")
st.markdown("Detecting anomalies using an Autoencoder trained on network intrusion data.")

# ────────────────────────────────────────────────
# File Upload Section
uploaded_file = st.file_uploader("📂 Upload your preprocessed CSV", type=["csv"])

# ────────────────────────────────────────────────
# Utility: Chunked MSE Computation
def batch_mse(input_data, model, batch_size=10000):
    mse_list = []
    for i in range(0, len(input_data), batch_size):
        batch = input_data[i:i+batch_size]
        
        # Sanity check: match model input shape
        if batch.shape[1] != model.input_shape[-1]:
            raise ValueError(f"❌ Shape mismatch: Model expects {model.input_shape[-1]} features, got {batch.shape[1]}")
        
        preds = model.predict(batch)
        batch_mse = np.mean(np.power(batch - preds, 2), axis=1)
        mse_list.extend(batch_mse)
    return np.array(mse_list)

# ────────────────────────────────────────────────
if uploaded_file is not None:
    # Step 1: Preview Data
    df = pd.read_csv(uploaded_file)
    st.write("### 🔎 Uploaded Data Preview", df.head())

    # Step 2: Preprocessing
    st.subheader("🧼 Data Preprocessing")
    df = df.select_dtypes(include=[np.number])  # Use only numeric data
    df.fillna(0, inplace=True)

    # Step 2.5: Enforce Expected Feature Count
    EXPECTED_FEATURES = 51  # Change this if your model was trained with a different count
    if df.shape[1] != EXPECTED_FEATURES:
        st.warning(f"⚠️ Expected {EXPECTED_FEATURES} features, but got {df.shape[1]}.")
        if st.checkbox("✅ Auto-trim to first 51 columns"):
            df = df.iloc[:, :EXPECTED_FEATURES]
            st.success(f"Trimmed to shape: {df.shape}")
        else:
            st.stop()

    # Step 2.6: Normalize
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df).astype(np.float32)
    st.write("📊 Normalized Data Sample", pd.DataFrame(df_scaled).head())

    # Step 3: Load Autoencoder
    st.subheader("📥 Loading Trained Autoencoder Model")
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "autoencoder_nids.h5"
    st.code(f"Model Path: {MODEL_PATH}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Autoencoder model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load model. Error: {e}")
        st.stop()

    # Step 4: Run Anomaly Detection
    st.subheader("🚨 Anomaly Detection in Progress")
    mse = batch_mse(df_scaled, model)

    # Step 5: Set Threshold
    default_thresh = float(np.percentile(mse, 95))
    threshold = st.slider(
        "📉 Adjust Detection Threshold (95th percentile default)",
        float(min(mse)), float(max(mse)),
        default_thresh, step=0.0001
    )

    # Step 6: Classify Anomalies
    is_anomaly = mse > threshold
    df_results = pd.DataFrame({
        "Reconstruction Error": mse,
        "Anomaly": is_anomaly
    })
    df_results["Label"] = df_results["Anomaly"].map({True: "🔺 Anomaly", False: "✅ Normal"})

    st.write("### 🧾 Detection Results", df_results.head())

    # Step 7: Visualization
    st.subheader("📈 Reconstruction Error Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(mse, bins=50, kde=True, ax=ax, color="mediumpurple")
    ax.axvline(threshold, color='red', linestyle='--', label='Threshold')
    ax.set_title("Histogram of Reconstruction Errors")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.legend()
    st.pyplot(fig)

    # Step 8: Downloadable CSV
    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Detection Results", csv, "detection_results.csv", "text/csv")
