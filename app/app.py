"""
This is the frontend for the port prediction model. 
This application takes in the following from the user:

    1. Anchor depth of the port (in meters)
    2. Cargo depth of the port (in meters)
    3. Distance to the nearest population center (in kilometers)
    4. Total amount of chinese debt of that country (millions or billions in USD)
    5. GDP of that country (millions or billions in USD)

The application will scale this data and run it against a logistic regression
model to determine whether or not the port meets the rough criteria
for China to potenially use for Naval purposes.
"""
import pickle
import streamlit as st
import pandas as pd
import numpy as np


# Loading the model and scaler
model_path = 'models/model.pkl'
scaler_path = 'models/scalers.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)


def input_data():
    st.header("Port and Country Parameters")

    # Create columns for better layout control
    col1, col2, col3 = st.columns(3)

    with col1:
        total_chinese_debt_value = st.number_input(
            "Total Chinese Debt (USD)", min_value=0.0, step=0.1)
        gdp_value = st.number_input("GDP (USD)", min_value=0.0, step=0.1)
        anchor_depth_m = st.number_input(
            "Anchor Depth (meters)", min_value=0.0, step=0.1)
        cargo_depth_m = st.number_input(
            "Cargo Depth (meters)", min_value=0.0, step=0.1)
        distance_to_nearest_population_center_km = st.number_input(
            "Distance to Nearest Population Center (km)", min_value=0.0, step=0.1)

    # Putting the dropdowns in a separate column so it looks a bit cleaner.
    with col2:
        total_chinese_debt_unit = st.selectbox(
            "Unit", ["in millions", "in billions"], key="debt_unit")
        if total_chinese_debt_unit == "in billions":
            total_chinese_debt = total_chinese_debt_value * 1e9
        else:
            total_chinese_debt = total_chinese_debt_value * 1e6

        gdp_unit = st.selectbox(
            "Unit", ["in millions", "in billions"], key="gdp_unit")
        if gdp_unit == "in billions":
            gdp = gdp_value * 1e9
        else:
            gdp = gdp_value * 1e6

    input_data = {
        "anchor_depth_m": anchor_depth_m,
        "cargo_depth_m": cargo_depth_m,
        "distance_to_nearest_population_center_km": distance_to_nearest_population_center_km,
        "Total chinese debt": total_chinese_debt,
        "GDP in 2021": gdp
    }

    return input_data

# We're storing what little styling we have in a CSS file
# To keep things from getting too cluttered.


def output_results(prediction):
    st.header("Potential for Chinese Naval Interest:")
    if prediction >= 0.5:
        st.markdown('<p class="return true">TRUE</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="return false">FALSE</p>',
                    unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Portal",
        layout="wide"
    )

    with open("components/styles.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),
                    unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([1, 4, 1])

        with col1:
            st.image("app-images/idsg-logo.png", width=150)

        with col2:
            st.markdown("<h1 class='title-text'>Portal</h1>",
                        unsafe_allow_html=True)
            st.markdown(
                "<p class='title-text'>Port metrics predictor to assess potential PLAN interest</p>", unsafe_allow_html=True)

        with col3:
            st.image("app-images/innovation-logo2.png", width=150)

    st.divider()

    with st.container():
        col1, col2 = st.columns([4, 1])

    # This is where Streamlit is irritating. We need to save
    # the prediction in state so we can pass it to col2. For
    # some reason each column has its own scope.
    with col1:
        user_input = input_data()
        if st.button("Generate Prediction"):
            input_df = pd.DataFrame([user_input])
            scaled_data = scaler.transform(input_df)
            prediction = model.predict_proba(scaled_data)[:, 1][0]
            st.session_state.prediction = prediction

    with col2:
        if "prediction" in st.session_state:
            output_results(st.session_state.prediction)


if __name__ == '__main__':
    main()
