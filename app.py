import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import cv2
import os
from PIL import Image
from docx import Document
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# Streamlit UI with branding
st.set_page_config(page_title="Chiller Optimization by Lavkosh Chavhan", layout="wide")
st.title("Chiller Energy Optimization & IPLV Prediction")
st.subheader("Developed by Lavkosh Chavhan")

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    """Extracts text from a Word document."""
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def preprocess_data(df):
    """Cleans and preprocesses the dataset, mapping human-readable column names to standardized ones."""
    column_mappings = {
        "Chilled Water Inlet Temp": "CHW_Inlet_Temp",
        "Chilled Water Outlet Temp": "CHW_Outlet_Temp",
        "Ambient Temperature": "Ambient_Temp",
        "Power Consumption (kW)": "Power_kW",
        "Flow Rate (GPM)": "Flow_Rate_GPM",
        "Cooling Load (kW)": "Cooling_kW"
    }
    
    df.columns = df.columns.str.strip()
    df.rename(columns={key: val for key, val in column_mappings.items() if key in df.columns}, inplace=True)
    df = df.dropna()
    
    if all(col in df.columns for col in ["CHW_Inlet_Temp", "CHW_Outlet_Temp", "Flow_Rate_GPM"]):
        df['CHW_Temp_Diff'] = df['CHW_Inlet_Temp'] - df['CHW_Outlet_Temp']
        df['Cooling_kW'] = df['Flow_Rate_GPM'] * 500 * df['CHW_Temp_Diff'] / 3412  # Convert to kW cooling
    
    required_columns = ["Power_kW", "Cooling_kW"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    
    return df

def train_model(df):
    """Trains a regression model to predict IPLV part-load efficiency."""
    features = ["CHW_Inlet_Temp", "CHW_Outlet_Temp", "Ambient_Temp", "Power_kW", "Cooling_kW"]
    target = "IPLV_Part_Load"
    
    if target not in df.columns:
        st.error("Missing target column: IPLV_Part_Load")
        return None, None, None
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2

def detect_anomalies(df):
    """Detects anomalies in power consumption using Isolation Forest."""
    if all(col in df.columns for col in ["Power_kW", "Cooling_kW"]):
        model = IsolationForest(contamination=0.05, random_state=42)
        df['Anomaly'] = model.fit_predict(df[["Power_kW", "Cooling_kW"]])
    else:
        st.error("Anomaly detection failed: Required columns are missing.")
    return df

uploaded_file = st.file_uploader("Upload Chiller Data (CSV/XLSX/PDF/DOCX)", type=["csv", "xlsx", "pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith("pdf"):
        text = extract_text_from_pdf(uploaded_file)
        st.write("Extracted Text from PDF:", text)
    elif uploaded_file.name.endswith("docx"):
        text = extract_text_from_docx(uploaded_file)
        st.write("Extracted Text from Word Document:", text)
    else:
        st.error("Unsupported file format")
    
    if 'df' in locals():
        df = preprocess_data(df)
        model, mae, r2 = train_model(df)
        df = detect_anomalies(df)
        
        st.write("### Model Performance")
        if model:
            st.write(f"Mean Absolute Error: {mae:.2f}")
            st.write(f"R-squared Score: {r2:.2f}")
        
        st.write("### Anomalies Detected")
        st.dataframe(df[df['Anomaly'] == -1] if "Anomaly" in df.columns else "No anomalies detected.")
        
        st.write("### Full Processed Data")
        st.dataframe(df)
