import streamlit as st
import pandas as pd
import joblib

# Load the Random Forest model
model = joblib.load('depl_model.pkl')

# Function to make predictions
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Make prediction
    prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    return prediction[0]

# Streamlit UI
def main():
    st.title('Crop Recommendation')
    st.sidebar.header('Input Parameters')

    # Get input values
    N = st.sidebar.slider('Nitrogen (N)', 0.0, 150.0, 50.0)
    P = st.sidebar.slider('Phosphorous (P)', 0.0, 100.0, 30.0)
    K = st.sidebar.slider('Potassium (K)', 0.0, 200.0, 70.0)
    temperature = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 25.0)
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 60.0)
    ph = st.sidebar.slider('pH', 0.0, 14.0, 7.0)
    rainfall = st.sidebar.slider('Rainfall (mm)', 0.0, 400.0, 100.0)

    if st.sidebar.button('Recommend Crop'):
        # Make prediction
        prediction = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        # Reverse transform the label (assuming you have a list of crop names)
        crops = ['rice', 'wheat', 'maize', 'millet', 'sugarcane', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbeans', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee', 'tea']
        crop = crops[prediction]
        st.write(f'Based on the input parameters, the recommended crop is: {crop}')

if __name__ == '__main__':
    main()
