import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and feature names
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

def preprocess_input(area, user_rating, total_ratings, region, rating_summary, hotel_category):
    input_data = pd.DataFrame(columns=feature_names)

    # Fill in the input data
    input_data.at[0, 'User Rating'] = user_rating
    input_data.at[0, 'Total Ratings'] = total_ratings

    # One-hot encode categorical variables
    categorical_features = {
        'Area': area,
        'Region': region,
        'Rating_Summary': rating_summary,
        'Hotel Category': hotel_category
    }
    for feature, value in categorical_features.items():
        input_data[f'{feature}_{value}'] = 1

    # Ensure all features are present in the input data
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    return input_data

# Streamlit app
st.title('OYO Room Price Prediction')

# Inputs from the user
area = st.text_input('Area')
user_rating = st.number_input('User Rating', min_value=0.0, max_value=5.0, step=0.1)
total_ratings = st.number_input('Total Ratings', min_value=0, step=1)
region = st.selectbox('Region', ['Chennai', 'Bangalore', 'Hyderabad', 'Mumbai', 'Ahmedabad', 'Jaipur', 'Delhi', 'Noida'])
rating_summary = st.selectbox('Rating Summary', ['Excellent', 'Good', 'Average', 'Poor', 'NEW'])
hotel_category = st.selectbox('Hotel Category', ['Super OYO', 'Collection O', 'Capital O', 'Flagship', 'Townhouse', 'OYO Homes', 'Spot ON', 'Silver Key', 'OYO Hotels', 'OYO Palatte'])

if st.button('Predict'):
    input_data = preprocess_input(area, user_rating, total_ratings, region, rating_summary, hotel_category)

    # Scale the input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Predict the discounted price
    predicted_price = model.predict(input_data_scaled)

    # Display the result
    st.write(f'The predicted discounted price for the OYO room is: ₹{predicted_price[0]:.2f}')
