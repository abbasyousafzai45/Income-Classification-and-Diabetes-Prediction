
# Import required libraries
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Cache the dataset so it loads only once and speeds up the Streamlit app
@st.cache_data
def load_data():
    # Load Iris dataset from sklearn
    iris = load_iris()
    
    # Convert the dataset into a pandas DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Add target column (species encoded as 0, 1, 2)
    df['species'] = iris.target
    
    # Return DataFrame and target class names
    return df, iris.target_names

# Call the function and store returned values
df, target_names = load_data()

# Create a Random Forest classification model
model = RandomForestClassifier()

# Train the model using feature columns (X) and target column (y)
model.fit(df.iloc[:, :-1], df['species'])

# Iris Flower Classification Title
st.title("Iris Flower Classification")

# Sidebar title in Streamlit UI
st.sidebar.title("Input Features")

# Sliders for user input (feature values)
sepal_length = st.sidebar.slider(
    "Sepal length",
    float(df['sepal length (cm)'].min()),
    float(df['sepal length (cm)'].max())
)

sepal_width = st.sidebar.slider(
    "Sepal width",
    float(df['sepal width (cm)'].min()),
    float(df['sepal width (cm)'].max())
)

petal_length = st.sidebar.slider(
    "Petal length",
    float(df['petal length (cm)'].min()),
    float(df['petal length (cm)'].max())
)

petal_width = st.sidebar.slider(
    "Petal width",
    float(df['petal width (cm)'].min()),
    float(df['petal width (cm)'].max())
)

# Combine user input into a single list (model expects 2D array)
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make prediction using trained model
prediction = model.predict(input_data)

# Convert numeric prediction into species name
predicted_species = target_names[prediction[0]]

# Display prediction result on the Streamlit app
st.write("Prediction")
st.write(f"The predicted species is: {predicted_species}")


