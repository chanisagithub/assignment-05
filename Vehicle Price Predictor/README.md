# Sri Lanka Vehicle Price Predictor

A web application to predict vehicle prices in Sri Lanka based on vehicle features using machine learning.

## Features

- Interactive web interface
- Real-time price predictions
- Support for multiple vehicle makes and models
- Considers vehicle age, mileage, engine capacity, and features
- Built with XGBoost machine learning model

## Installation

1. Install required dependencies:
```bash
pip install streamlit pandas numpy joblib xgboost scikit-learn
```

## Running the Application

Navigate to the application folder and run:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. Select the vehicle make (brand)
2. Set the model popularity (how common the model is)
3. Enter year of manufacture
4. Input mileage and engine capacity
5. Select transmission type and fuel type
6. Check optional features (AC, Power Steering, Power Mirrors)
7. Click "Predict Price" to get the estimated value

## Model Information

- Algorithm: XGBoost Regressor
- Training Data: Vehicle listings from riyasewana.com
- Features: 22 input features including vehicle specifications and features
- Target: Vehicle price in Sri Lankan Rupees

## Note

Predictions are estimates based on historical data. Actual market prices may vary based on:
- Vehicle condition
- Location
- Market demand
- Seller motivation
