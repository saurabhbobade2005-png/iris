
import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load('saurabh_moadel.pkl')

# Page title
st.title('Iris Flower Classification')

# Input labels
sepal_length = st.number_input('Sepal Length (cm)')
sepal_width = st.number_input('Sepal Width (cm)')
petal_length = st.number_input('Petal Length (cm)')
petal_width = st.number_input('Petal Width (cm)')

# Prediction
if st.button('Predict'):
  input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1,-1)
  prediction = model.predict(input_data)
  st.write(f"The predicted species is: {prediction[0]}")
