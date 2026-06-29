import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/house_price_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("🏠 Предсказание цены дома")
st.write("Введите параметры дома, чтобы получить предсказанную цену")

overall_qual = st.slider("Общее качество дома (1-10)", 1, 10, 5)
gr_liv_area = st.number_input(
    "Жилая площадь (кв. футы)", min_value=300, max_value=6000, value=1500
)
garage_cars = st.slider("Количество машин в гараже", 0, 4, 2)
total_bsmt_sf = st.number_input(
    "Площадь подвала (кв. футы)", min_value=0, max_value=3000, value=800
)
year_built = st.number_input(
    "Год постройки", min_value=1900, max_value=2024, value=2000
)
full_bath = st.slider("Количество полных ванных", 0, 4, 2)

if st.button("Предсказать цену"):
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

    input_data["OverallQual"] = overall_qual
    input_data["GrLivArea"] = gr_liv_area
    input_data["GarageCars"] = garage_cars
    input_data["TotalBsmtSF"] = total_bsmt_sf
    input_data["YearBuilt"] = year_built
    input_data["FullBath"] = full_bath

    log_prediction = model.predict(input_data)[0]
    price_prediction = np.expm1(log_prediction)

    st.success(f"Предсказанная цена: ${price_prediction:,.0f}")
st.markdown(
    """
<style>
.stButton button {
    background-color: #ff4b4b;
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)
