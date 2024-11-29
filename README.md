# Housing-Price-Prediction
Using Linear Regression to predict housing price
# House Price Prediction Streamlit App

## Overview
This Streamlit application predicts house prices using a Linear Regression model trained on housing dataset features.

## Features
- Interactive web interface for house price prediction
- Machine learning model with various input parameters
- Real-time prediction and model performance metrics
- Comprehensive data visualizations

## Prerequisites
- Python 3.8+
- Libraries: 
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - seaborn
  - matplotlib

## Installation
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

## Dataset
- Uses `Housing.csv`
- Features include area, bedrooms, bathrooms, location characteristics

## Model Details
- Algorithm: Linear Regression
- Preprocessing: 
  - Categorical variable encoding
  - Feature scaling
- Evaluation Metric: R-squared Score

## How to Run
```bash
streamlit run app.py
```

## App Sections
1. Input Parameters Sidebar
   - Adjust house features interactively
2. Prediction Display
   - Shows predicted house price
3. Model Performance
   - Displays R-squared score
4. Data Visualizations
   - Correlation Heatmap
   - Price Distribution
   - Area vs Price Scatter Plot
   - Feature Importance Bar Chart

## Customization
- Modify `housing.py` to add more features
- Experiment with different machine learning models


## Author
Pranamya Deshpande
