import streamlit as st
import pandas as pd
import joblib

# Loading wallpaper
def set_bg_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function with direct image URL
set_bg_url('https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8YWZyaWNhbiUyMHdpbGRsaWZlfGVufDB8fDB8fHww')


st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load('SalesML_Model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error("Error: Could not load 'SalesML_Model.joblib'. Ensure the model file is in this directory.")
    st.stop()

# Central input layout using 2 columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    total_spent = st.number_input("Total Spent ($)", min_value=0.0, value=1500.0, step=50.0)

with col2:
    support_tickets = st.number_input("Support Tickets Logged", min_value=0, max_value=50, value=2, step=1)
    satisfaction_score = st.number_input("Satisfaction Score (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

st.divider()

# Prediction trigger
if st.button("Predict Churn", type="primary", use_container_width=True):
    # Construct raw DataFrame
    raw_input = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'total_spent': total_spent,
        'support_tickets': support_tickets,
        'satisfaction_score': satisfaction_score
    }])

    # Preprocess & align columns to model features
    input_encoded = pd.get_dummies(raw_input, columns=['gender'], drop_first=True)
    expected_cols = getattr(model, 'feature_names_in_', input_encoded.columns)
    ready_input = input_encoded.reindex(columns=expected_cols, fill_value=0)

    # Output prediction
    prediction = model.predict(ready_input)[0]
    
    if prediction == 1:
        st.error("Result: High Churn Risk (Customer likely to leave)")
    else:
        st.success("Result: Low Churn Risk (Customer likely to stay)")