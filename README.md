# ML Tasks: Income & Diabetes Prediction

This Streamlit application provides interactive interfaces for two machine learning tasks:

1. Income Prediction (Task 1)
   - Predicts whether an individual's income exceeds $50K/year
   - Features: age, education, occupation, marital status, hours worked per week, etc.
   - Uses Random Forest classifier with optional hyperparameter tuning

2. Diabetes Prediction (Task 2)
   - Predicts diabetes diagnosis based on medical measurements
   - Features: pregnancies, BMI, blood pressure, insulin level, age, etc.
   - Uses both Decision Tree and Random Forest classifiers with tuning options

## Setup & Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Using the App

1. Select the task from the sidebar (Income or Diabetes prediction)
2. Run the pipeline to train models
3. Use the manual prediction form to input features and get predictions
4. Optionally tune models using sidebar buttons for better performance

Note: The first run may take longer as it processes and caches the data.
