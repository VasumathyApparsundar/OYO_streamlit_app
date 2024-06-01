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
    feature_names = list(feature_names)  # Convert Index object to list

# Streamlit app
st.title('OYO Room Price Prediction')

# Inputs from the user
area = st.text_input('Area')
user_rating = st.number_input('User Rating', min_value=0.0, max_value=5.0, step=0.1)
total_ratings = st.number_input('Total Ratings', min_value=0, step=1)
region = st.selectbox('Region', ['Chennai', 'Bangalore', 'Hyderabad', 'Mumbai', 'Ahmedabad', 'Jaipur', 'Delhi', 'Noida'])
rating_summary = st.selectbox('Rating Summary', ['Excellent', 'Good', 'Average', 'Poor', 'NEW'])
hotel_category = st.selectbox('Hotel Category', ['Super OYO', 'Collection O', 'Capital O', 'Flagship', 'Townhouse', 'OYO Homes', 'Spot ON', 'Silver Key', 'OYO Hotels', 'OYO Palatte'])

# Create a predict button
if st.button('Predict'):
    # Preprocess the inputs
    def preprocess_input(area, user_rating, total_ratings, region, rating_summary, hotel_category):
        # Initialize input_data with zeros
        input_data = np.zeros(len(feature_names))
        
        # Map categorical variables to their corresponding columns in input_data
        # Area
        if area in feature_names:
            input_data[feature_names.index(area)] = 1
        # User Rating and Total Ratings
        input_data[feature_names.index('User Rating')] = user_rating
        input_data[feature_names.index('Total Ratings')] = total_ratings
        # Region
        if f'Region_{region}' in feature_names:
            input_data[feature_names.index(f'Region_{region}')] = 1
        # Rating Summary
        if f'Rating_Summary_{rating_summary}' in feature_names:
            input_data[feature_names.index(f'Rating_Summary_{rating_summary}')] = 1
        # Hotel Category
        if f'Hotel Category_{hotel_category}' in feature_names:
            input_data[feature_names.index(f'Hotel Category_{hotel_category}')] = 1
        
        return input_data

    # Get the preprocessed input_data
    input_data = preprocess_input(area, user_rating, total_ratings, region, rating_summary, hotel_category)
    
    # Ensure input_data has the same number of features as the model expects
    if len(input_data) != len(feature_names):
        st.error(f"Number of features in input data ({len(input_data)}) does not match the expected number of features ({len(feature_names)})")
    else:
        # Reshape input_data to match model's expected input shape
        input_data = input_data.reshape(1, -1)

        # Scale the input data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        # Predict the discounted price
        predicted_price = model.predict(input_data_scaled)

        # Display the result
        st.write(f'The predicted discounted price for the OYO room is: ₹{predicted_price[0]:.2f}')
