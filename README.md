# Housing-Price-Prediction
Using Hybrid Model(Linear Regression and XGBOOST) to predict Housing Price.
Click here to try: https://housing-price-prediction-pranamya-deshpande.streamlit.app
# House Price Prediction Streamlit App

## Overview
A machine learning application for predicting house prices using Linear Regression and XGBoost models with an interactive Streamlit interface.

## Features
- Interactive house price prediction
- Multiple machine learning models (Linear Regression, XGBoost, Ensemble)
- Real-time price prediction
- Comprehensive data visualizations
- Model performance metrics

## Prerequisites
- Python 3.8+

## Installation
```bash
git clone https://github.com/Pranamya16/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

## Dataset
- Uses `Housing.csv`
- Features include:
  - Area
  - Bedrooms
  - Bathrooms
  - Location characteristics
  - Amenities (mainroad, guestroom, basement, etc.)

## Model Details
- Algorithms: 
  - Linear Regression
  - XGBoost
  - Ensemble Model (Weighted Average)
- Preprocessing: 
  - Categorical variable encoding
  - Feature scaling
- Evaluation Metric: R-squared Score

## How to Run
```bash
streamlit run housing.py
```

## App Sections
1. Input Parameters Sidebar
2. Price Prediction
3. Model Performance Metrics
4. Data Visualizations
   - Correlation Heatmap
   - Price Distribution
   - Area vs Price Scatter Plot
   - Model Prediction Plots
   - Feature Importance Charts

## Customization
- Modify code to add more features
- Experiment with model parameters

## Author
Pranamya Deshpande
