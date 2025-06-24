import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = tf.keras.models.load_model(r"C:\Users\Sheraz\Documents\pythontest\project_all_ml\VCodez\autoencoder_model.h5")

# Function to normalize input data
def preprocess_input(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled

# Streamlit UI
st.title("IoT Sensor Anomaly Detection")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    scaled_data = preprocess_input(df)
    predictions = model.predict(scaled_data)

    reconstruction_error = np.mean(np.abs(scaled_data - predictions), axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    anomalies = reconstruction_error > threshold

    df["Anomaly"] = anomalies
    st.write("Anomaly Detection Results:", df)
