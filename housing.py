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


# train the model
model = LinearRegression()
model.fit(x_train_scaled, y_train)


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

# make prediction
prediction = model.predict(input_scaled)

st.subheader("Prediction")
st.write(f"The predicted House Price is: ${prediction[0]:,.2f}")

# model performance
y_pred = model.predict(x_test_scaled)
mse = mean_squared_error(
    y_test, y_pred
)  # ytest is the expected answer and ypred is the predicted answer
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"R-sqared Score: {r2:.2f}")

corr = data.corr()
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


st.write("Price Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(data["price"], kde=True, color="red", edgecolor="black")
plt.title("Price Distribution")
plt.xlabel("Price", fontsize=12)
plt.ylabel("Count", fontsize=12)
st.pyplot(plt)

# Scatter plot: Area vs Price with regression line
st.write("Area vs Price")
plt.figure(figsize=(10, 6))
sns.regplot(
    x="area",
    y="price",
    data=data,
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
)
plt.title("Area vs Price", fontsize=16)
plt.xlabel("Area", fontsize=12)
plt.ylabel("Price", fontsize=12)
st.pyplot(plt)

# Feature importance (updated for Linear Regression)
st.write("Important Features for Predicting Price")
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
