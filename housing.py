import streamlit as st  # web app framework
import pandas as pd  # for data manipulation (reading dataset)
import numpy as np  # for nbumerical operations
import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
from sklearn.model_selection import (
    train_test_split,
)  # splits data into training and testing parts
from sklearn.preprocessing import StandardScaler  # used to scale in range 0-1
from sklearn.metrics import mean_squared_error, r2_score  # count the cost error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

st.set_page_config(
    page_title="House Price Prediction",
)


# load the data
@st.cache_data
def load_data():
    data = pd.read_csv("Housing.csv")
    return data


data = load_data()  # loading data in the variable 'data'


# Preprocessing the data
def preprocess_data(df):
    # Convert categorical variables to numeric
    df["mainroad"] = df["mainroad"].map({"yes": 1, "no": 0})
    df["guestroom"] = df["guestroom"].map({"yes": 1, "no": 0})
    df["basement"] = df["basement"].map({"yes": 1, "no": 0})
    df["hotwaterheating"] = df["hotwaterheating"].map({"yes": 1, "no": 0})
    df["airconditioning"] = df["airconditioning"].map({"yes": 1, "no": 0})
    df["prefarea"] = df["prefarea"].map({"yes": 1, "no": 0})
    df["furnishingstatus"] = df["furnishingstatus"].map(
        {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    )
    return df


data = preprocess_data(data)

# Split the data
x = data.drop("price", axis=1)
y = data["price"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# scaler the features in the range 0-1
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# streamlit app

st.title("House Price Prediction")
st.caption("By Pranamya Deshpande")

# User input
st.sidebar.header("Input Parameters")


def user_input_features():
    area = st.sidebar.slider(
        "Area", float(x["area"].min()), float(x["area"].max()), float(x["area"].mean())
    )
    bedrooms = st.sidebar.slider(
        "Bedrooms",
        int(x["bedrooms"].min()),
        int(x["bedrooms"].max()),
        int(x["bedrooms"].mean()),
    )
    bathrooms = st.sidebar.slider(
        "Bathrooms",
        int(x["bathrooms"].min()),
        int(x["bathrooms"].max()),
        int(x["bathrooms"].mean()),
    )
    stories = st.sidebar.slider(
        "Stories",
        int(x["stories"].min()),
        int(x["stories"].max()),
        int(x["stories"].mean()),
    )
    mainroad = st.sidebar.selectbox("Mainroad", ("yes", "no"))
    guestroom = st.sidebar.selectbox("Guest Room", ("yes", "no"))
    basement = st.sidebar.selectbox("Basement", ("yes", "no"))
    hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ("yes", "no"))
    airconditioning = st.sidebar.selectbox("Air Conditioning", ("yes", "no"))
    parking = st.sidebar.slider(
        "Parking",
        int(x["parking"].min()),
        int(x["parking"].max()),
        int(x["parking"].mean()),
    )
    prefarea = st.sidebar.selectbox("Preferred Area", ("yes", "no"))
    furnishingstatus = st.sidebar.selectbox(
        "Furnishing Status", ("furnished", "semi-furnished", "unfurnished")
    )
    # dictionary where keys are the featurenames and the values are user input
    features = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }
    return pd.DataFrame(features, index=[0])


input_df = user_input_features()
inpu_df = preprocess_data(input_df)  # convert categories into binary values

# scale user input
input_scaled = scaler.transform(input_df)

# train the LR model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Train XGBoost Model
model_xgb = xgb.XGBRegressor(random_state=42)
model_xgb.fit(x_train_scaled, y_train)

# Ensemble Prediction Function
def ensemble_prediction(input_scaled):
    lr_pred = model.predict(input_scaled)
    xgb_pred = model_xgb.predict(input_scaled)
    
    # Weighted average (60% XGBoost, 40% Linear Regression)
    ensemble_pred = 0.6 * xgb_pred + 0.4 * lr_pred
    return ensemble_pred

prediction = ensemble_prediction(input_scaled)

# Add to model performance section
# Predict on test set
y_pred_lr = model.predict(x_test_scaled)
y_pred_xgb = model_xgb.predict(x_test_scaled)
y_pred_ensemble = ensemble_prediction(x_test_scaled)
corr = data.corr()

# Compute additional metrics
st.subheader("Prediction")
st.write(f"The predicted House Price is: ${prediction[0]:,.2f}")
st.subheader("Model Performance Comparison")
st.write(f"Linear Regression R-squared: {r2_score(y_test, y_pred_lr):.2f}")
st.write(f"XGBoost R-squared: {r2_score(y_test, y_pred_xgb):.2f}")
st.write(f"Ensemble R-squared: {r2_score(y_test, y_pred_ensemble):.2f}")

# data Visualization
st.subheader("Data visualization")

# setting seaborn style for DV
sns.set_style("whitegrid")
sns.set_palette("deep")

# correllation heatmap
st.write("Corelation Heatmap")
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f",
    square=True,
)
plt.title("Correlatin Heatmap", fontsize=16)
st.pyplot(plt)

# Create visualization columns
viz_col1, viz_col2, viz_col3 = st.columns(3)
# Linear Regression Plot
with viz_col1:
    st.write("Linear Regression Predictions")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred_lr, color='blue', line_kws={'color': 'red'})
    plt.title('Linear Regression Predictions')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    st.pyplot(plt)

# XGBoost Plot
with viz_col2:   
    st.write("XGBoost Predictions")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred_xgb, color='green', line_kws={'color': 'red'})
    plt.title('XGBoost Predictions')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    st.pyplot(plt)

# Ensemble Model Plot
with viz_col3:
    st.write("Ensemble Predictions")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred_ensemble, color='purple', line_kws={'color': 'red'})
    plt.title('Ensemble Model Predictions')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    st.pyplot(plt)

# Add this to the existing Streamlit script, after the existing model training and prediction code

# Scatter plot comparing model predictions
st.write("Model Prediction Comparison")
plt.figure(figsize=(12, 6))

# Create a DataFrame with actual and predicted values
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Linear Regression': y_pred_lr,
    'XGBoost': y_pred_xgb,
    'Ensemble': y_pred_ensemble
})

# Melt the DataFrame for easier plotting with seaborn
comparison_df_melted = comparison_df.melt(id_vars='Actual', 
                                          var_name='Model', 
                                          value_name='Predicted')

# Create a scatter plot with different colors for each model
sns.scatterplot(data=comparison_df_melted, 
                x='Actual', 
                y='Predicted', 
                hue='Model', 
                palette='deep')

# Add a perfect prediction line
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', 
         label='Perfect Prediction')

plt.title("Model Prediction Comparison", fontsize=16)
plt.xlabel("Actual House Prices", fontsize=12)
plt.ylabel("Predicted House Prices", fontsize=12)
plt.legend(title='Model Types')
st.pyplot(plt)

# Feature importance (updated for Linear Regression)
st.write("Linear Regression Feature Importance")
feature_importance = pd.DataFrame(
    {
        "feature": x.columns,
        "importance": np.abs(model.coef_),  # Use absolute values of coefficients
    }
)
feature_importance = feature_importance.sort_values("importance", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feature_importance, color="steelblue")
plt.title("Feature Importance for Price Prediction", fontsize=16)
plt.xlabel("Absolute Coefficient Value", fontsize=12)
plt.ylabel("Feature", fontsize=12)
st.pyplot(plt)


# XGBoost Feature Importance
st.write("XGBoost Feature Importance")
xgb_importance = pd.DataFrame({
    'feature': x.columns,
    'importance': model_xgb.feature_importances_
})
xgb_importance = xgb_importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=xgb_importance, color='red')
plt.title("XGBoost Feature Importance", fontsize=16)
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
st.pyplot(plt)
