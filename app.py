import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    layout='wide', 
    page_title='Diabetes Prediction System',
    page_icon='🩺',
    initial_sidebar_state='expanded'
)

# Advanced preprocessing function for diabetes prediction
def advanced_diabetes_preprocessing(df):
    """Enhanced preprocessing with feature engineering"""
    df_processed = df.copy()
    
    # Handle zero values more intelligently
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        zero_mask = df_processed[column] == 0
        if zero_mask.sum() > 0:
            # Use median of non-zero values for each outcome class
            for outcome in [0, 1]:
                class_mask = df_processed['Outcome'] == outcome
                zero_class_mask = zero_mask & class_mask
                if zero_class_mask.sum() > 0:
                    median_val = df_processed[class_mask & ~zero_mask][column].median()
                    df_processed.loc[zero_class_mask, column] = median_val
    
    # Feature Engineering - Create new meaningful features
    df_processed['BMI_Category'] = pd.cut(df_processed['BMI'], 
                                        bins=[0, 18.5, 25, 30, float('inf')], 
                                        labels=[0, 1, 2, 3])
    
    df_processed['Glucose_Level'] = pd.cut(df_processed['Glucose'], 
                                         bins=[0, 100, 125, float('inf')], 
                                         labels=[0, 1, 2])
    
    df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                     bins=[0, 30, 40, 50, float('inf')], 
                                     labels=[0, 1, 2, 3])
    
    # Interaction features
    df_processed['BMI_Glucose_Interaction'] = df_processed['BMI'] * df_processed['Glucose']
    df_processed['Age_BMI_Interaction'] = df_processed['Age'] * df_processed['BMI']
    
    # Ratio features
    df_processed['Glucose_BMI_Ratio'] = df_processed['Glucose'] / (df_processed['BMI'] + 1)
    
    return df_processed

# Initialize diabetes model and data
@st.cache_data
def load_and_train_diabetes_model():
    """Load data and train diabetes model once, cache for performance"""
    
    # Load diabetes dataset
    diabetes_df = pd.read_csv('diabetes.csv')
    
    # Enhanced Diabetes Model Training
    diabetes_enhanced = advanced_diabetes_preprocessing(diabetes_df)
    X_diabetes = diabetes_enhanced.drop('Outcome', axis=1)
    y_diabetes = diabetes_enhanced['Outcome']
    
    # Split and prepare diabetes data
    X_train_diab, X_test_diab, y_train_diab, y_test_diab = train_test_split(
        X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
    )
    
    # Feature selection for diabetes model
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train_diab, y_train_diab)
    X_test_selected = selector.transform(X_test_diab)
    selected_features = X_diabetes.columns[selector.get_support()]
    
    # Scale diabetes features
    diabetes_scaler = RobustScaler()
    X_train_scaled = diabetes_scaler.fit_transform(X_train_selected)
    X_test_scaled = diabetes_scaler.transform(X_test_selected)
    
    # Train multiple models and create ensemble
    models = {}
    
    # SVM Model
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, 
                   class_weight='balanced', random_state=42)
    svm_model.fit(X_train_scaled, y_train_diab)
    models['SVM'] = svm_model
    
    # Enhanced Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                    class_weight='balanced', random_state=42)
    rf_model.fit(X_train_scaled, y_train_diab)
    models['RandomForest'] = rf_model
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                        learning_rate=0.1, random_state=42)
    gb_model.fit(X_train_scaled, y_train_diab)
    models['GradientBoosting'] = gb_model
    
    # Create ensemble
    ensemble_model = VotingClassifier([
        ('svm', svm_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ], voting='soft')
    ensemble_model.fit(X_train_scaled, y_train_diab)
    
    # Evaluate ensemble
    ensemble_accuracy = ensemble_model.score(X_test_scaled, y_test_diab)
    y_pred = ensemble_model.predict(X_test_scaled)
    ensemble_recall = recall_score(y_test_diab, y_pred)
    ensemble_f1 = f1_score(y_test_diab, y_pred)
    ensemble_precision = precision_score(y_test_diab, y_pred)
    
    return {
        'diabetes_model': ensemble_model,
        'diabetes_scaler': diabetes_scaler,
        'feature_selector': selector,
        'selected_features': selected_features,
        'diabetes_features': list(selected_features),
        'diabetes_accuracy': ensemble_accuracy,
        'diabetes_recall': ensemble_recall,
        'diabetes_f1': ensemble_f1,
        'diabetes_precision': ensemble_precision,
        'model_type': 'Advanced Ensemble (SVM + RF + GB)'
    }

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, 
                    insulin, bmi, diabetes_pedigree, age, models_data):
    """Enhanced diabetes prediction with advanced preprocessing"""
    
    # Create input dataframe
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Add dummy outcome for preprocessing
    input_data['Outcome'] = 0
    
    # Apply advanced preprocessing
    processed_data = advanced_diabetes_preprocessing(input_data)
    
    # Select features and scale
    X_input = processed_data.drop('Outcome', axis=1)
    X_selected = models_data['feature_selector'].transform(X_input)
    X_scaled = models_data['diabetes_scaler'].transform(X_selected)
    
    # Predict using ensemble model
    prediction = models_data['diabetes_model'].predict(X_scaled)[0]
    probability = models_data['diabetes_model'].predict_proba(X_scaled)[0, 1]
    
    return prediction, probability

# Load diabetes model
@st.cache_resource
def get_diabetes_model():
    return load_and_train_diabetes_model()

# Main App - Diabetes Only
def main():
    # Load diabetes model
    models_data = get_diabetes_model()
    
    # App header
    st.title('🩺 Advanced Diabetes Prediction System')
    st.markdown('### Enhanced Diabetes Risk Assessment with SVM + Random Forest + Gradient Boosting')
    
    # Model performance info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{models_data['diabetes_accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{models_data['diabetes_precision']:.1%}")
    with col3:
        st.metric("Recall", f"{models_data['diabetes_recall']:.1%}")
    with col4:
        st.metric("F1-Score", f"{models_data['diabetes_f1']:.3f}")
    
    # Model details
    with st.expander("🔍 Advanced Model Details"):
        st.write("**Diabetes Prediction Model:**")
        st.write(f"- Algorithm: {models_data['model_type']}")
        st.write(f"- Selected Features: {len(models_data['selected_features'])} features")
        st.write("- Advanced preprocessing with feature engineering")
        st.write("- Ensemble voting for improved accuracy")
        st.write("- Optimized for medical diagnosis (high sensitivity)")
        
        st.write("**Selected Features:**")
        for i, feature in enumerate(models_data['selected_features']):
            st.write(f"{i+1}. {feature}")
    
    st.header("🩺 Diabetes Risk Assessment")
    st.success("🚀 **Using Advanced ML Model** - SVM + Random Forest + Gradient Boosting Ensemble")
    st.markdown("**Comprehensive diabetes risk analysis using advanced machine learning**")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Basic Information")
        pregnancies = st.number_input("Number of Pregnancies", 0, 17, 0, 
                                    help="Total number of pregnancies")
        age = st.slider("Age (years)", 21, 81, 33, 
                      help="Patient age in years")
        
        st.subheader("🩸 Laboratory Results")
        glucose = st.slider("Glucose Level (mg/dL)", 0, 199, 120,
                          help="Plasma glucose concentration (normal: 70-100 mg/dL)")
        blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 122, 70,
                                 help="Diastolic blood pressure (normal: <80 mmHg)")
        
    with col2:
        st.subheader("📏 Physical Measurements")
        bmi = st.slider("BMI (Body Mass Index)", 0.0, 67.1, 32.0, 0.1,
                      help="Body Mass Index (normal: 18.5-24.9)")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20,
                                 help="Triceps skin fold thickness")
        
        st.subheader("🧬 Medical History")
        insulin = st.slider("Insulin Level (μU/mL)", 0, 846, 79,
                          help="2-hour serum insulin (normal: 16-166 μU/mL)")
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, 0.001,
                                    help="Diabetes pedigree function (genetic predisposition)")
    
    # Risk indicators
    with st.expander("📊 Medical Risk Indicators (Auto-calculated)"):
        col3, col4 = st.columns(2)
        with col3:
            # BMI category
            if bmi < 18.5:
                bmi_cat = "🔵 Underweight"
            elif bmi < 25:
                bmi_cat = "🟢 Normal Weight"
            elif bmi < 30:
                bmi_cat = "🟡 Overweight"
            else:
                bmi_cat = "🔴 Obese"
            st.write(f"**BMI Category:** {bmi_cat}")
            
            # Glucose category
            if glucose < 100:
                glucose_cat = "🟢 Normal"
            elif glucose < 125:
                glucose_cat = "🟡 Prediabetic Range"
            else:
                glucose_cat = "🔴 Diabetic Range"
            st.write(f"**Glucose Level:** {glucose_cat}")
            
        with col4:
            # Blood pressure category
            if blood_pressure < 80:
                bp_cat = "🟢 Normal"
            elif blood_pressure < 90:
                bp_cat = "🟡 High Normal"
            else:
                bp_cat = "🔴 High"
            st.write(f"**Blood Pressure:** {bp_cat}")
            
            # Age risk
            if age < 35:
                age_risk = "🟢 Low Risk"
            elif age < 45:
                age_risk = "🟡 Moderate Risk"
            else:
                age_risk = "🔴 High Risk"
            st.write(f"**Age Risk:** {age_risk}")
    
    # Real-time prediction
    if st.button("🚀 Analyze Diabetes Risk", type="primary", use_container_width=True):
        with st.spinner("Performing advanced ML analysis..."):
            prediction, probability = predict_diabetes(
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age, models_data
            )
            
            # Enhanced result display
            st.markdown("---")
            st.subheader("📋 Advanced Analysis Results")
            
            # Main prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error(f"⚠️ **HIGH RISK: Diabetes Detected**")
                    st.error(f"**Medical Assessment: POSITIVE**")
                else:
                    st.success(f"✅ **LOW RISK: No Diabetes Detected**")
                    st.success(f"**Medical Assessment: NEGATIVE**")
            
            with col2:
                confidence = max(probability, 1-probability)
                st.metric("🎯 Diagnostic Confidence", f"{confidence:.1%}")
                st.metric("📈 Diabetes Probability", f"{probability:.1%}")
            
            with col3:
                # Risk level categorization
                if probability > 0.8:
                    risk_level = "🔴 VERY HIGH RISK"
                elif probability > 0.6:
                    risk_level = "🟠 HIGH RISK"
                elif probability > 0.4:
                    risk_level = "🟡 MODERATE RISK"
                elif probability > 0.2:
                    risk_level = "🔵 LOW RISK"
                else:
                    risk_level = "🟢 VERY LOW RISK"
                
                st.metric("🚨 Risk Level", risk_level)
            
            # Progress bar for risk level
            st.progress(probability)
            
            # Detailed risk assessment
            st.subheader("🔍 Detailed Medical Assessment")
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.write("**🔺 Risk Factors Present:**")
                risk_factors = []
                
                if glucose > 125:
                    risk_factors.append("• **Hyperglycemia** - Diabetic glucose levels")
                elif glucose > 100:
                    risk_factors.append("• **Prediabetes** - Impaired glucose tolerance")
                
                if bmi > 30:
                    risk_factors.append("• **Obesity** - Significant diabetes risk factor")
                elif bmi > 25:
                    risk_factors.append("• **Overweight** - Moderate risk factor")
                
                if blood_pressure > 90:
                    risk_factors.append("• **Hypertension** - Cardiovascular risk")
                
                if age > 45:
                    risk_factors.append("• **Advanced Age** - Age-related diabetes risk")
                
                if pregnancies >= 4:
                    risk_factors.append("• **Multiple Pregnancies** - Gestational diabetes history risk")
                
                if diabetes_pedigree > 0.5:
                    risk_factors.append("• **Strong Genetic Predisposition** - Family history")
                
                if insulin > 200:
                    risk_factors.append("• **Insulin Resistance** - Metabolic dysfunction")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.write("• No significant risk factors detected")
            
            with col5:
                st.write("**🔻 Protective Factors:**")
                protective_factors = []
                
                if glucose < 100:
                    protective_factors.append("• **Normal Glucose** - Healthy glucose metabolism")
                
                if bmi >= 18.5 and bmi < 25:
                    protective_factors.append("• **Healthy Weight** - Optimal BMI range")
                
                if blood_pressure < 80:
                    protective_factors.append("• **Normal Blood Pressure** - Good cardiovascular health")
                
                if age < 35:
                    protective_factors.append("• **Young Age** - Lower diabetes risk")
                
                if diabetes_pedigree < 0.3:
                    protective_factors.append("• **Low Genetic Risk** - Minimal family history")
                
                if insulin >= 16 and insulin <= 166:
                    protective_factors.append("• **Normal Insulin** - Good insulin sensitivity")
                
                if protective_factors:
                    for factor in protective_factors:
                        st.write(factor)
                else:
                    st.write("• Limited protective factors identified")
            
            # Medical recommendations
            st.subheader("💡 Medical Recommendations")
            
            if probability > 0.7:
                st.error("🚨 **URGENT MEDICAL CONSULTATION REQUIRED**")
                st.write("**Immediate Actions:**")
                st.write("• Schedule endocrinologist appointment within 24-48 hours")
                st.write("• Request comprehensive diabetes panel (HbA1c, fasting glucose)")
                st.write("• Begin glucose monitoring if advised by physician")
                st.write("• Implement immediate dietary changes")
                
            elif probability > 0.5:
                st.warning("⚠️ **HIGH RISK - Medical Consultation Recommended**")
                st.write("**Recommended Actions:**")
                st.write("• Schedule appointment with primary care physician")
                st.write("• Request diabetes screening tests")
                st.write("• Begin lifestyle modifications (diet and exercise)")
                st.write("• Monitor symptoms and risk factors closely")
                
            elif probability > 0.3:
                st.info("🟡 **MODERATE RISK - Preventive Measures**")
                st.write("**Preventive Actions:**")
                st.write("• Annual diabetes screening recommended")
                st.write("• Maintain healthy diet and regular exercise")
                st.write("• Weight management if BMI >25")
                st.write("• Regular health monitoring")
                
            else:
                st.success("✅ **LOW RISK - Continue Healthy Lifestyle**")
                st.write("**Maintenance Actions:**")
                st.write("• Continue current healthy habits")
                st.write("• Regular health check-ups")
                st.write("• Diabetes screening every 3 years after age 45")
                st.write("• Monitor for any changes in health status")
            
            # Model performance info
            with st.expander("🔬 Advanced Model Performance Details"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{models_data['diabetes_accuracy']:.1%}")
                with col2:
                    st.metric("Precision", f"{models_data['diabetes_precision']:.1%}")
                with col3:
                    st.metric("Recall (Sensitivity)", f"{models_data['diabetes_recall']:.1%}")
                with col4:
                    st.metric("F1-Score", f"{models_data['diabetes_f1']:.3f}")
                
                st.write("**Advanced Model Features:**")
                st.write("- **SVM with RBF kernel** for non-linear pattern recognition")
                st.write("- **Random Forest** with 200 trees for robust predictions")
                st.write("- **Gradient Boosting** for sequential error correction")
                st.write("- **Ensemble voting** combining all three algorithms")
                st.write("- **Advanced feature engineering** (BMI categories, interactions, ratios)")
                st.write("- **Class balancing** for optimal medical diagnosis")
                st.write("- **Feature selection** using statistical tests")
                st.write("- **Robust scaling** for consistent performance")
    
    # Performance info
    st.markdown("---")
    st.info("🚀 **Advanced ML Architecture:** This system uses SVM + Random Forest + Gradient Boosting ensemble with advanced feature engineering for superior medical accuracy.")
    st.caption("⚠️ **Medical Disclaimer:** This AI tool is designed to assist healthcare professionals and should not replace clinical judgment. Always consult qualified healthcare providers for medical decisions and diagnosis.")

if __name__ == "__main__":
    main()
