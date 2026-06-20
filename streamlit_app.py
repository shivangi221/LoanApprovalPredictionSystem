import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Loan Approval Prediction System", page_icon="🏦")

# Load the saved model and preprocessor
@st.cache_resource
def load_artifacts():
    try:
        # Use Path to get the directory of the current script
        script_dir = Path(__file__).parent
        
        try:
            xgb_model = joblib.load(script_dir / 'xgb_model.joblib')
        except Exception as e:
            st.error(f"Failed to load xgb_model: {e}")
            st.stop()
            
        try:
            preprocessor = joblib.load(script_dir / 'preprocessor.joblib')
        except Exception as e:
            st.error(f"Failed to load preprocessor (version mismatch?): {e}. Ensure sklearn version matches training environment.")
            st.stop()
            
        try:
            le_target = joblib.load(script_dir / 'le_target.joblib')
        except Exception as e:
            st.error(f"Failed to load le_target: {e}")
            st.stop()
            
        try:
            important_feature_names = joblib.load(script_dir / 'important_feature_names.joblib')
        except Exception as e:
            st.error(f"Failed to load important_feature_names: {e}")
            st.stop()
            
        return xgb_model, preprocessor, le_target, important_feature_names
    except Exception as e:
        st.error(f"Model or preprocessing artifacts not found: {e}. Please ensure all .joblib files are in the same directory.")
        st.stop()

xgb_model, preprocessor, le_target, important_feature_names = load_artifacts()

# Handle input resets by clearing session state keys
if 'reset_inputs' not in st.session_state:
    st.session_state.reset_inputs = False

if st.session_state.reset_inputs:
    keys_to_reset = [
        'applicant_income', 'age', 'marital_status', 'profession',
        'coapplicant_income', 'family_size', 'education_level', 'employment_status',
        'credit_score', 'existing_loans', 'dti_ratio', 'gender',
        'loan_amount', 'loan_term', 'loan_purpose', 'property_area',
        'dependents', 'collateral_value', 'savings', 'employer_category'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.reset_inputs = False

st.title('🏦 Loan Approval Prediction System')
st.markdown('Enter applicant details to predict loan approval status. We use an XGBoost model trained on various applicant and financial features.')

# --- Sidebar for General Info or Less Critical Inputs ---
st.sidebar.header('About the Application')
st.sidebar.info("This application uses a pre-trained machine learning model to predict loan approval. The model considers several factors to make a decision.")

st.sidebar.header('Model Details')
st.sidebar.write(f"**Model Type:** XGBoost Classifier")
st.sidebar.write(f"**Features Used:** {len(important_feature_names)} important features")

# --- Input Fields in Main Area, Organized into Columns ---
st.header('Applicant Details')

col1, col2, col3 = st.columns(3)

with col1:
    applicant_income = st.number_input('Applicant Income (USD)', min_value=0.0, value=50000.0, step=1000.0, help="Gross monthly income of the applicant.", key='applicant_income')
    age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1, help="Age of the loan applicant.", key='age')
    marital_status = st.selectbox('Marital Status', ['Married', 'Single', 'Divorced', 'Widowed'], help="Applicant's marital status.", key='marital_status')
    profession = st.selectbox('Profession', ['Salaried', 'Self-employed', 'Government', 'Business', 'Other'], help="Applicant's profession.", key='profession')

with col2:
    coapplicant_income = st.number_input('Coapplicant Income (USD)', min_value=0.0, value=20000.0, step=1000.0, help="Gross monthly income of the coapplicant, if any.", key='coapplicant_income')
    family_size = st.number_input('Family Size', min_value=0, value=2, step=1, help="Number of dependents in the applicant's family.", key='family_size')
    education_level = st.selectbox('Education Level', ['Graduate', 'Not Graduate'], help="Applicant's highest education level.", key='education_level')
    employment_status = st.selectbox('Employment Status', ['Salaried', 'Self-employed', 'Government', 'Unemployed', 'Contract'], help="Current employment status.", key='employment_status')

with col3:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=700, step=1, help="Applicant's credit score (e.g., FICO).", key='credit_score')
    existing_loans = st.number_input('Number of Existing Loans', min_value=0, value=1, step=1, help="Total number of active loans the applicant currently has.", key='existing_loans')
    dti_ratio = st.number_input('Debt-to-Income Ratio', min_value=0.0, max_value=1.0, value=0.4, format="%.2f", help="Monthly debt payments divided by gross monthly income.", key='dti_ratio')
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'], help="Applicant's gender.", key='gender')

st.subheader('Loan Details')
loan_col1, loan_col2, loan_col3 = st.columns(3)

with loan_col1:
    loan_amount = st.number_input('Loan Amount (USD)', min_value=1000.0, value=100000.0, step=5000.0, key='loan_amount')
    loan_term = st.number_input('Loan Term (in months)', min_value=12, max_value=360, value=120, step=12, key='loan_term')

with loan_col2:
    loan_purpose = st.selectbox('Loan Purpose', ['Home', 'Education', 'Car', 'Business', 'Other'], key='loan_purpose')
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'], help="Location of property", key='property_area')
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'], help="Number of dependents", key='dependents')

with loan_col3:
    collateral_value = st.number_input('Collateral Value (USD)', min_value=0.0, value=150000.0, step=10000.0, key='collateral_value')
    savings = st.number_input('Savings (USD)', min_value=0.0, value=10000.0, step=1000.0, key='savings')
    employer_category = st.selectbox('Employer Category', ['Private', 'Government', 'Business', 'MNC', 'Self-employed', 'Unemployed', 'Other'], key='employer_category')

# Predict button
if st.button('Predict Loan Approval', type='primary'):
    with st.spinner('Making a prediction...'):
        # Create a DataFrame from the input data
        input_data = pd.DataFrame({
            'Applicant_Income': [applicant_income],
            'Coapplicant_Income': [coapplicant_income],
            'Employment_Status': [employment_status],
            'Age': [age],
            'Marital_Status': [marital_status],
            'Education_Level': [education_level],
            'Family_Size': [family_size],
            'Dependents': [dependents],
            'Profession': [profession],
            'Credit_Score': [credit_score],
            'Loan_Term': [loan_term],
            'Existing_Loans': [existing_loans],
            'DTI_Ratio': [dti_ratio],
            'Property_Area': [property_area],
            'Savings': [savings],
            'Collateral_Value': [collateral_value],
            'Loan_Amount': [loan_amount],
            'Loan_Purpose': [loan_purpose],
            'Gender': [gender],
            'Employer_Category': [employer_category]
        })

        # Re-create the engineered feature (must be consistent with training)
        if 'Loan_Amount' in input_data.columns and 'Applicant_Income' in input_data.columns:
            input_data['Loan_Amount_to_Income'] = input_data['Loan_Amount'] / (input_data['Applicant_Income'] + 1e-6)

        # Preprocess the input data using the loaded preprocessor
        processed_input = preprocessor.transform(input_data)

        # Apply feature selection using the saved important_feature_names
        processed_feature_names_list = preprocessor.get_feature_names_out().tolist()
        processed_input_df = pd.DataFrame(processed_input, columns=processed_feature_names_list)

        # Filter important_feature_names to ensure they are present in processed_feature_names_list
        actual_important_features = [f for f in important_feature_names if f in processed_feature_names_list]
        if not actual_important_features:
            st.error("No matching important features found for prediction after preprocessing.")
            st.stop()

        input_for_prediction = processed_input_df[actual_important_features]

        # Make prediction
        prediction_numeric = xgb_model.predict(input_for_prediction)[0]
        prediction_proba = xgb_model.predict_proba(input_for_prediction)[0]

        # Decode the prediction back to 'Yes'/'No'
        prediction_label = le_target.inverse_transform([prediction_numeric])[0]

        st.subheader('Prediction Result:')
        if prediction_label == 'Yes':
            st.success(f'🎉 Loan Approval Status: **{prediction_label}**')
            st.metric(label="Confidence (Probability of Yes)", value=f"{prediction_proba[le_target.transform(['Yes'])[0]]:.2%}")
        else:
            st.error(f'❌ Loan Approval Status: **{prediction_label}**')
            st.metric(label="Confidence (Probability of No)", value=f"{prediction_proba[le_target.transform(['No'])[0]]:.2%}")

        st.markdown("--- Disclaimer ---")
        st.info("Note: This is a model prediction and should not be used as the sole basis for real-world loan decisions. Always consult with financial experts.")

# Reset button
if st.button('Reset Inputs', type='secondary'):
    st.session_state.reset_inputs = True
    st.rerun()

st.caption("Developed with Streamlit and scikit-learn.")