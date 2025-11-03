import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

st.set_page_config(
    layout='wide', 
    page_title='Real-Time ML Predictions',
    page_icon='🔮',
    initial_sidebar_state='expanded'
)

# Initialize models and data
@st.cache_data
def load_and_train_models():
    """Load data and train models once, cache for performance"""
    
    # Load datasets
    adult_df = pd.read_csv('adult.csv')
    diabetes_df = pd.read_csv('diabetes.csv')
    
    # Preprocess Income Data
    adult_df = adult_df.replace('?', np.nan)
    adult_df = adult_df.dropna(subset=['workclass', 'occupation', 'native.country'])
    if 'fnlwgt' in adult_df.columns:
        adult_df = adult_df.drop('fnlwgt', axis=1)
    adult_df['income'] = adult_df['income'].str.strip()
    adult_df['income_encoded'] = (adult_df['income'] == '>50K').astype(int)
    
    # Income features
    num_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    num_cols = [c for c in num_cols if c in adult_df.columns]
    cat_cols = [c for c in adult_df.columns if adult_df[c].dtype == 'object' and c != 'income']
    cat_df = pd.get_dummies(adult_df[cat_cols], drop_first=True)
    
    X_income = pd.concat([adult_df[num_cols].reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    y_income = adult_df['income_encoded']
    
    # Train Income Model
    X_train_inc, X_test_inc, y_train_inc, y_test_inc = train_test_split(
        X_income, y_income, test_size=0.2, random_state=42, stratify=y_income
    )
    income_model = RandomForestClassifier(n_estimators=100, random_state=42)
    income_model.fit(X_train_inc, y_train_inc)
    
    # Preprocess Diabetes Data
    diabetes_df = diabetes_df.copy()
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        if col in diabetes_df.columns:
            median_val = diabetes_df[diabetes_df[col] != 0][col].median()
            diabetes_df[col] = diabetes_df[col].replace(0, median_val)
    
    X_diabetes = diabetes_df.drop('Outcome', axis=1)
    y_diabetes = diabetes_df['Outcome']
    
    # Train Diabetes Model
    X_train_diab, X_test_diab, y_train_diab, y_test_diab = train_test_split(
        X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
    )
    
    # Scale diabetes features
    diabetes_scaler = StandardScaler()
    X_train_diab_scaled = diabetes_scaler.fit_transform(X_train_diab)
    X_test_diab_scaled = diabetes_scaler.transform(X_test_diab)
    
    diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_model.fit(X_train_diab_scaled, y_train_diab)
    
    # Get categorical options for income prediction
    cat_options = {}
    for col in cat_cols:
        if col in adult_df.columns:
            cat_options[col] = sorted(adult_df[col].dropna().unique().tolist())
    
    return {
        'income_model': income_model,
        'diabetes_model': diabetes_model,
        'diabetes_scaler': diabetes_scaler,
        'income_features': X_income.columns.tolist(),
        'diabetes_features': X_diabetes.columns.tolist(),
        'cat_options': cat_options,
        'income_accuracy': income_model.score(X_test_inc, y_test_inc),
        'diabetes_accuracy': diabetes_model.score(X_test_diab_scaled, y_test_diab)
    }

def predict_income(age, workclass, education, marital_status, occupation, 
                  relationship, race, sex, capital_gain, capital_loss, 
                  hours_per_week, native_country, models_data):
    """Real-time income prediction"""
    
    # Create input dataframe
    input_data = {
        'age': age,
        'education.num': education,
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
        'hours.per.week': hours_per_week
    }
    
    # Categorical data
    cat_data = {
        'workclass': workclass,
        'education': education,
        'marital.status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'native.country': native_country
    }
    
    # Create dataframes
    num_df = pd.DataFrame([input_data])
    cat_df = pd.DataFrame([cat_data])
    cat_dummies = pd.get_dummies(cat_df, drop_first=True)
    
    # Combine and align with training features
    input_row = pd.concat([num_df.reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)
    input_row = input_row.reindex(columns=models_data['income_features'], fill_value=0)
    
    # Predict
    prediction = models_data['income_model'].predict(input_row)[0]
    probability = models_data['income_model'].predict_proba(input_row)[0, 1]
    
    return prediction, probability

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, 
                    insulin, bmi, diabetes_pedigree, age, models_data):
    """Real-time diabetes prediction"""
    
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]], columns=models_data['diabetes_features'])
    
    # Scale the input
    input_scaled = models_data['diabetes_scaler'].transform(input_data)
    
    # Predict
    prediction = models_data['diabetes_model'].predict(input_scaled)[0]
    probability = models_data['diabetes_model'].predict_proba(input_scaled)[0, 1]
    
    return prediction, probability

# Load models and data
@st.cache_resource
def get_models():
    return load_and_train_models()

# Main App
def main():
    # Load models once
    models_data = get_models()
    
    # App header
    st.title('🔮 Real-Time ML Predictions')
    st.markdown('### Instant Income & Diabetes Predictions')
    
    # Model accuracy info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Income Model Accuracy", f"{models_data['income_accuracy']:.1%}")
    with col2:
        st.metric("Diabetes Model Accuracy", f"{models_data['diabetes_accuracy']:.1%}")
    
    # Sidebar for prediction type
    st.sidebar.title("🎯 Choose Prediction")
    prediction_type = st.sidebar.radio(
        "Select what to predict:",
        ["💰 Income Prediction", "🩺 Diabetes Prediction"]
    )
    
    if prediction_type == "💰 Income Prediction":
        st.header("💰 Income Prediction")
        st.markdown("**Will this person earn more than $50K annually?**")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Info")
            age = st.slider("Age", 17, 90, 39)
            sex = st.selectbox("Sex", models_data['cat_options']['sex'])
            race = st.selectbox("Race", models_data['cat_options']['race'])
            relationship = st.selectbox("Relationship", models_data['cat_options']['relationship'])
            
        with col2:
            st.subheader("💼 Work & Education")
            workclass = st.selectbox("Work Class", models_data['cat_options']['workclass'])
            education = st.selectbox("Education", models_data['cat_options']['education'])
            occupation = st.selectbox("Occupation", models_data['cat_options']['occupation'])
            hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        
        # Additional info in expandable section
        with st.expander("💵 Financial Details (Optional)"):
            col3, col4 = st.columns(2)
            with col3:
                capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
                marital_status = st.selectbox("Marital Status", models_data['cat_options']['marital.status'])
            with col4:
                capital_loss = st.number_input("Capital Loss", 0, 4356, 0)
                native_country = st.selectbox("Native Country", models_data['cat_options']['native.country'])
        
        # Real-time prediction
        if st.button("🔮 Predict Income", type="primary"):
            with st.spinner("Predicting..."):
                # Map education to education.num (simplified mapping)
                education_num_map = {
                    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
                    '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
                    'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
                    'Prof-school': 15, 'Doctorate': 16
                }
                education_num = education_num_map.get(education, 10)
                
                prediction, probability = predict_income(
                    age, workclass, education_num, marital_status, occupation,
                    relationship, race, sex, capital_gain, capital_loss,
                    hours_per_week, native_country, models_data
                )
                
                # Display result with styling
                if prediction == 1:
                    st.success(f"💰 **Prediction: Income > $50K**")
                    st.success(f"🎯 **Confidence: {probability:.1%}**")
                else:
                    st.info(f"💼 **Prediction: Income ≤ $50K**")
                    st.info(f"🎯 **Confidence: {1-probability:.1%}**")
                
                # Progress bar for probability
                st.progress(probability)
    
    elif prediction_type == "🩺 Diabetes Prediction":
        st.header("🩺 Diabetes Risk Assessment")
        st.markdown("**Does this patient have diabetes?**")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Basic Info")
            pregnancies = st.number_input("Pregnancies", 0, 17, 0)
            age = st.slider("Age", 21, 81, 33)
            
            st.subheader("🩸 Blood Tests")
            glucose = st.slider("Glucose Level", 0, 199, 120)
            blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
            
        with col2:
            st.subheader("📏 Physical Measurements")
            bmi = st.slider("BMI", 0.0, 67.1, 32.0, 0.1)
            skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
            
            st.subheader("🧬 Medical History")
            insulin = st.slider("Insulin Level", 0, 846, 79)
            diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, 0.001)
        
        # Real-time prediction
        if st.button("🔮 Predict Diabetes Risk", type="primary"):
            with st.spinner("Analyzing..."):
                prediction, probability = predict_diabetes(
                    pregnancies, glucose, blood_pressure, skin_thickness,
                    insulin, bmi, diabetes_pedigree, age, models_data
                )
                
                # Display result with styling
                if prediction == 1:
                    st.error(f"⚠️ **High Risk: Diabetes Detected**")
                    st.error(f"🎯 **Risk Level: {probability:.1%}**")
                else:
                    st.success(f"✅ **Low Risk: No Diabetes**")
                    st.success(f"🎯 **Confidence: {1-probability:.1%}**")
                
                # Risk level indicator
                if probability > 0.7:
                    st.markdown("🚨 **Recommendation:** Immediate medical consultation advised")
                elif probability > 0.4:
                    st.markdown("⚠️ **Recommendation:** Regular monitoring suggested")
                else:
                    st.markdown("✅ **Recommendation:** Continue healthy lifestyle")
                
                # Progress bar for risk level
                st.progress(probability)

if __name__ == "__main__":
    main()
