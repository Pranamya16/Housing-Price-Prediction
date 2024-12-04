import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Performance Optimization: Reduce page config overhead
st.set_page_config(page_title="House Price Prediction", layout="wide")

# Caching data loading for improved performance
@st.cache_data
def load_and_preprocess_data():
    # Load and preprocess data in a single cached function
    data = pd.read_csv("Housing.csv")
    
    # Preprocessing
    categorical_cols = [
        "mainroad", "guestroom", "basement", "hotwaterheating", 
        "airconditioning", "prefarea"
    ]
    for col in categorical_cols:
        data[col] = (data[col] == "yes").astype(int)
    
    data["furnishingstatus"] = data["furnishingstatus"].map({
        "furnished": 2, 
        "semi-furnished": 1, 
        "unfurnished": 0
    })
    
    return data

# Caching model training to avoid recomputation
@st.cache_resource
def train_models(x_train_scaled, y_train):
    # Train models with caching
    model_lr = LinearRegression()
    model_xgb = xgb.XGBRegressor(random_state=42)
    
    model_lr.fit(x_train_scaled, y_train)
    model_xgb.fit(x_train_scaled, y_train)
    
    return model_lr, model_xgb

def main():
    st.title("House Price Prediction")
    st.caption("By Pranamya Deshpande")

    # Load and preprocess data
    data = load_and_preprocess_data()

    # Prepare features and target
    x = data.drop("price", axis=1)
    y = data["price"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train models
    model_lr, model_xgb = train_models(x_train_scaled, y_train)

    # Create a session state to store user inputs
    if 'user_inputs' not in st.session_state:
        st.session_state.user_inputs = {}

    # User input section
    st.sidebar.header("Input House Parameters")
    
    def get_user_inputs():
        # Use session state to dynamically update inputs
        for col in x.columns:
            if col in ["mainroad", "guestroom", "basement", "hotwaterheating", 
                       "airconditioning", "prefarea"]:
                st.session_state.user_inputs[col] = st.sidebar.selectbox(
                    col.capitalize(), 
                    options=["yes", "no"],
                    key=f"{col}_select",
                    # Use previous value if exists
                    index=["yes", "no"].index(st.session_state.user_inputs.get(col, "no"))
                )
            elif col == "furnishingstatus":
                st.session_state.user_inputs[col] = st.sidebar.selectbox(
                    "Furnishing Status", 
                    options=["furnished", "semi-furnished", "unfurnished"],
                    key=f"{col}_select",
                    # Use previous value if exists
                    index=["furnished", "semi-furnished", "unfurnished"].index(
                        st.session_state.user_inputs.get(col, "unfurnished")
                    )
                )
            elif col in ["area", "bedrooms", "bathrooms", "stories", "parking"]:
                st.session_state.user_inputs[col] = st.sidebar.slider(
                    col.capitalize(), 
                    float(x[col].min()), 
                    float(x[col].max()), 
                    float(st.session_state.user_inputs.get(col, x[col].mean())),
                    key=f"{col}_slider"
                )
        
        return pd.DataFrame([st.session_state.user_inputs])

    # Get and preprocess user inputs
    input_df = get_user_inputs()
    
    # Preprocess input (convert to same format as training data)
    processed_input = load_and_preprocess_data().loc[0:0, x.columns].copy()
    for col, value in st.session_state.user_inputs.items():
        if col in ["mainroad", "guestroom", "basement", "hotwaterheating", 
                   "airconditioning", "prefarea"]:
            processed_input.loc[0, col] = int(value == "yes")
        elif col == "furnishingstatus":
            processed_input.loc[0, col] = int({
                "furnished": 2, 
                "semi-furnished": 1, 
                "unfurnished": 0
            }[value])
        else:
            processed_input.loc[0, col] = float(value)
    
    # Scale input
    input_scaled = scaler.transform(processed_input)

    # Ensemble Prediction
    def ensemble_prediction(input_scaled):
        lr_pred = model_lr.predict(input_scaled)
        xgb_pred = model_xgb.predict(input_scaled)
        return 0.6 * xgb_pred + 0.4 * lr_pred

    # Prediction
    prediction = ensemble_prediction(input_scaled)
    
    # Create a placeholder for dynamic prediction
    prediction_placeholder = st.empty()
    prediction_placeholder.subheader("Predicted House Price")
    prediction_placeholder.write(f"The predicted House Price is: ${prediction[0]:,.2f}")

    # Performance Metrics
    st.subheader("Model Performance")
    
    # Predict on test set
    y_pred_lr = model_lr.predict(x_test_scaled)
    y_pred_xgb = model_xgb.predict(x_test_scaled)
    y_pred_ensemble = ensemble_prediction(x_test_scaled)

    # Metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Linear Regression R²", f"{r2_score(y_test, y_pred_lr):.2f}")
    with col2:
        st.metric("XGBoost R²", f"{r2_score(y_test, y_pred_xgb):.2f}")
    with col3:
        st.metric("Ensemble R²", f"{r2_score(y_test, y_pred_ensemble):.2f}")

    # Visualization Section
    st.subheader("Detailed Model Insights")

    # Interactive Feature Importance
    def plot_feature_importance():
        combined_importance = pd.DataFrame({
            'Feature': x.columns,
            'Importance': 0.4 * np.abs(model_lr.coef_) + 0.6 * model_xgb.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            combined_importance, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Feature Importance Comparison"
        )
        st.plotly_chart(fig)

    # Interactive Scatter Plot
    def plot_model_comparison():
        fig = go.Figure()
        
        # Add scatter plots for each model
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred_lr, 
            mode='markers', 
            name='Linear Regression',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred_xgb, 
            mode='markers', 
            name='XGBoost',
            marker=dict(color='green', opacity=0.6)
        ))
        
        # Perfect prediction line
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()], 
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dot')
        ))
        
        fig.update_layout(
            title='Model Prediction Comparison',
            xaxis_title='Actual Prices',
            yaxis_title='Predicted Prices'
        )
        
        st.plotly_chart(fig)

    # Call visualization functions
    plot_feature_importance()
    plot_model_comparison()

# Run the app
if __name__ == "__main__":
    main()
