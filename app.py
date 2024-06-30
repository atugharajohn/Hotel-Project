import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Azure Blob Storage connection details
storage_account_name = "interviewtask2024"
storage_account_key = "TixNz4h7ObZgTmC9wP9kgsknPIRvh4dudaLd15wZ6CirLFKsfl7vaahnOy1CH2gQMzNv1rPh6myX+AStRJvCdg=="
container_name = "picklefiles"

# Function to download pickle file from Azure Blob Storage
def download_pickle_from_blob(storage_account_name, storage_account_key, container_name, blob_name):
    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=storage_account_key)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    stream = BytesIO()
    blob_client.download_blob().download_to_stream(stream)
    return stream.getvalue()

# Load the regression model
model_reg_blob_name = 'model_reg.pkl'  # Update with your blob file name for regression model
try:
    model_reg_bytes = download_pickle_from_blob(storage_account_name, storage_account_key, container_name, model_reg_blob_name)
    model_reg = joblib.load(BytesIO(model_reg_bytes))
except Exception as e:
    st.error(f"Error loading regression model from Azure Blob Storage: {e}")
    st.stop()

# Load the classification model
model_class_blob_name = 'model_class.pkl'  # Update with your blob file name for classification model
try:
    model_class_bytes = download_pickle_from_blob(storage_account_name, storage_account_key, container_name, model_class_blob_name)
    model_class = joblib.load(BytesIO(model_class_bytes))
except Exception as e:
    st.error(f"Error loading classification model from Azure Blob Storage: {e}")
    st.stop()

# Load the training columns for regression
train_columns_reg_blob_name = 'train_columns.pkl'  # Update with your blob file name for regression train columns
try:
    train_columns_reg_bytes = download_pickle_from_blob(storage_account_name, storage_account_key, container_name, train_columns_reg_blob_name)
    train_columns_reg = joblib.load(BytesIO(train_columns_reg_bytes))
except Exception as e:
    st.error(f"Error loading regression train columns from Azure Blob Storage: {e}")
    st.stop()

# Load the training columns for classification
train_columns_class_blob_name = 'train_columns_classification.pkl'  # Update with your blob file name for classification train columns
try:
    train_columns_class_bytes = download_pickle_from_blob(storage_account_name, storage_account_key, container_name, train_columns_class_blob_name)
    train_columns_class = joblib.load(BytesIO(train_columns_class_bytes))
except Exception as e:
    st.error(f"Error loading classification train columns from Azure Blob Storage: {e}")
    st.stop()

# Function to preprocess data for regression
def preprocess_data_regression(df, train_columns):
    # Handle missing values: mean for numeric columns, mode for categorical columns
    numeric_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                    'required_car_parking_space', 'lead_time', 'repeated_guest',
                    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                    'no_of_special_requests']

    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

    # Fill missing values
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, columns=categorical_cols)

    # Ensure columns are aligned with training data
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing dummy columns with default value

    # Ensure the order of columns is consistent
    df = df[train_columns]

    return df

# Function to preprocess data for classification
def preprocess_data_classification(df, train_columns):
    # Handle missing values: mean for numeric columns, mode for categorical columns
    numeric_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                    'required_car_parking_space', 'lead_time', 'repeated_guest',
                    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                    'no_of_special_requests']

    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

    # Fill missing values
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, columns=categorical_cols)

    # Ensure columns are aligned with training data
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing dummy columns with default value

    # Ensure the order of columns is consistent
    df = df[train_columns]

    return df

# Streamlit App
st.title('Hotel Prediction App')

# Upload CSV data for Regression
st.sidebar.title('Upload CSV File for Regression')
uploaded_file_reg = st.sidebar.file_uploader("Choose a CSV file for Regression", type="csv")

if uploaded_file_reg is not None:
    # Read the CSV file
    df_reg = pd.read_csv(uploaded_file_reg)

    # Preprocess the data for regression
    df_processed_reg = preprocess_data_regression(df_reg, train_columns_reg)

    # Make regression predictions
    predictions_reg = model_reg.predict(df_processed_reg)

    # Display regression results
    st.write("### Input Data (Regression):")
    st.write(df_reg)

    # Add actual price and Booking_ID alongside Regression Predictions
    results_reg = pd.DataFrame({
        'Booking_ID': df_reg['Booking_ID'],
        'Actual_Price': df_reg['avg_price_per_room'],  # Replace 'actual_price_column' with your actual column name
        'Predicted_Price': predictions_reg
    })
    st.write("### Regression Predictions:")
    st.write(results_reg)


# Upload CSV data for Classification
st.sidebar.title('Upload CSV File for Classification')
uploaded_file_classification = st.sidebar.file_uploader("Choose a CSV file for Classification", type="csv")

if uploaded_file_classification is not None:
    # Read the CSV file
    df_classification = pd.read_csv(uploaded_file_classification)

    # Preprocess the data for classification
    df_processed_classification = preprocess_data_classification(df_classification, train_columns_class)

    # Make classification predictions
    predictions_classification = model_class.predict(df_processed_classification)

    # Display classification results
    st.write("### Input Data (Classification):")
    st.write(df_classification)

    # Add actual status and Booking_ID alongside Classification Predictions
    results_classification = pd.DataFrame({
        'Booking_ID': df_classification['Booking_ID'],
        'Actual_Status': df_classification['booking_status'],  # Replace 'actual_status_column' with your actual column name
        'Predicted_Status': predictions_classification
    })
    st.write("### Classification Predictions:")
    st.write(results_classification)
