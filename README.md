# 🚀 Anomaly-Based Network Intrusion Detection System (NIDS)

This project implements an **Anomaly-Based Intrusion Detection System (IDS)** using **Unsupervised Machine Learning** to detect malicious network traffic. The system is trained on the **CICIDS2017 dataset**, and uses an **autoencoder model** to identify anomalies in network behavior.

---

## 📌 Features
- ✅ Preprocessing of CICIDS2017 dataset (cleaning, normalization, feature selection)  
- ✅ Unsupervised learning with **Autoencoder Neural Network**  
- ✅ Detection of anomalies in network traffic  
- ✅ Model evaluation with accuracy, precision, recall, and F1-score  
- ✅ Modular project structure (data, models, scripts, reports)  

---

## 📂 Project Structure
```

Anomaly-NIDS-Project/
├── data/                   # Raw and processed dataset (ignored in Git)
├── models/                 # Saved models and training checkpoints
├── scripts/                # Python scripts for training, evaluation, detection
│   ├── train\_autoencoder.py
│   ├── evaluate\_model.py
│   └── detect\_anomalies.py
├── autoencoder\_model.h5    # Final trained Autoencoder model
├── requirements.txt        # Python dependencies
├── report.docx/            # Project report
├── README.md               # Project documentation
└── .gitignore

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Anomaly-NIDS-Project.git
cd Anomaly-NIDS-Project
````

### 2️⃣ Create a virtual environment

```bash
python -m venv nids_env
nids_env\Scripts\activate   # On Windows
source nids_env/bin/activate  # On Linux/Mac
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

We use the **CICIDS2017 dataset** provided by the Canadian Institute for Cybersecurity (CIC).
📌 [Dataset Link](https://www.unb.ca/cic/datasets/ids-2017.html)

Since the dataset is large, it is **ignored in Git**. Please download and place it under the `data/` directory before running experiments.

---

## 🚀 Usage

### Train the Autoencoder

```bash
python scripts/train_autoencoder.py
```

### Evaluate the Model

```bash
python scripts/evaluate_model.py
```

### Detect Anomalies

```bash
python scripts/detect_anomalies.py
```

---

## 📈 Results

* The trained autoencoder successfully detects anomalies in network traffic.
* Performance metrics (example):

  * Accuracy: \~95%
  * Precision: \~93%
  * Recall: \~92%
  * F1-score: \~92%

(Values may vary depending on preprocessing and hyperparameters.)

---

## 🛠️ Tech Stack

* **Languages**: Python
* **Libraries**: NumPy, Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib
* **Dataset**: CICIDS2017
* **Environment**: VS Code (Python virtual environment)

---

## 📑 Report

A detailed project report is available in [`report.docx/`](report.docx/).

---

## ✨ Future Work

* Deploy IDS as a real-time monitoring service.
* Extend to handle additional datasets and attack types.
* Integrate visualization dashboard for anomaly alerts.

---

## 👩‍💻 Author

**Lalithanjali Aruna Bikkavolu**
🔗 [LinkedIn](https://www.linkedin.com/in/lalithanjali-arun-bikkavolu-361090270/) | 🔗 [GitHub](https://github.com/lalithanjaliaruna123)
