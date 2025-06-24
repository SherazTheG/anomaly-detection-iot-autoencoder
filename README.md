
# 📡 Anomaly Detection in IoT Sensor Data using Autoencoders

This project uses a TensorFlow-based Autoencoder model to detect anomalies in IoT sensor data. A clean and interactive **Streamlit frontend** is provided for easy use. The original dataset is sourced from **Kaggle** and contains environmental sensor readings.

---

## 🗂️ Dataset Information

- **Source:** Kaggle (IoT Sensor Dataset)
- **Original Features:** `Time`, `Temperature`, `Humidity`, `Air Quality`, `Light`, `Loudness`
- **Used After Cleaning:** `Temperature`, `Humidity`, `Air Quality`, `Light`, `Loudness`

The goal is to detect unusual readings that may indicate sensor failure, tampering, or real-world anomalies.

---

## 🧠 Model Overview

- **Architecture:** Autoencoder
- **Framework:** TensorFlow
- **Training:** Trained manually using the cleaned dataset
- **Input Normalization:** Min-Max Scaling (`sklearn.preprocessing.MinMaxScaler`)
- **Anomaly Threshold:** 95th percentile of reconstruction error

---

## 🖥️ Streamlit App

### 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anomaly-detection-iot-autoencoder.git
cd anomaly-detection-iot-autoencoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run App.py
```

### 🧾 Input

- Upload a cleaned `.csv` file with the following columns:
  ```
  Temperature, Humidity, Air Quality, Light, Loudness
  ```

### 📊 Output

- Displays:
  - Preview of your uploaded data
  - Detected anomalies
  - Adds a new column: `Anomaly` (True/False)

---

## 📁 Project Structure

```
anomaly-detection-iot-autoencoder/
├── App.py                  # Streamlit frontend
├── dataset.csv             # Dataset
├── dataset_cleaned.csv     # Cleaned Dataset
├── Preliminary Report.docx # Basic Document Overview
├── Project.ipynb           # Model Building
├── autoencoder_model.h5    # Trained TensorFlow model
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 👤 Author

**Mohamed Sheraz**

---

## 📌 Notes

- This project was developed as part of a machine learning internship training.
- Anomalies are defined as entries whose reconstruction error exceeds the 95th percentile.

---

## 📜 License

This project is for educational and demonstration purposes.
