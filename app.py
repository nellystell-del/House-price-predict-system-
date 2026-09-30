import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rwanda House Price Prediction", page_icon="🏠", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("house_price_model_nelly_stella.sav")

model=load_model()

st.title("🏠 Rwanda House Price Prediction")
st.write("Enter house characteristics to estimate the price in Million RWF.")

c1,c2=st.columns(2)
with c1:
    area=st.number_input("Area (m²)",min_value=1.0,value=150.0,step=1.0)
    bedrooms=st.number_input("Bedrooms",min_value=1,value=3,step=1)
    bathrooms=st.number_input("Bathrooms",min_value=1,value=2,step=1)
    age=st.number_input("House Age (Years)",min_value=0.0,value=5.0,step=1.0)
with c2:
    distance=st.number_input("Distance to City Centre (km)",min_value=0.0,value=5.0,step=.1)
    parking=st.number_input("Parking Spaces",min_value=0,value=2,step=1)
    neighborhood=st.selectbox("Neighborhood",[
        'Gasabo',
        'Huye',
        'Kicukiro',
        'Kigali City',
        'Musanze',
        'Nyarugenge'
    ])

if st.button("Predict House Price",type="primary",use_container_width=True):
    house=pd.DataFrame({
        "Area_m2":[area],"Bedrooms":[bedrooms],"Bathrooms":[bathrooms],
        "House_Age_Years":[age],"Distance_to_City_km":[distance],
        "Parking_Spaces":[parking],"Neighborhood":[neighborhood]
    })
    prediction=float(model.predict(house)[0])
    st.success("Prediction completed.")
    st.metric("Estimated House Price",f"{prediction:,.2f} Million RWF")
    st.dataframe(house,use_container_width=True)
    result=house.copy()
    result["Predicted_Price_Million_RWF"]=prediction
    st.download_button("Download Prediction CSV",result.to_csv(index=False),
                       "nelly_stella_house_prediction.csv","text/csv",
                       use_container_width=True)

st.caption("Rwanda House Price Prediction — Nelly Stella Nzeyimana")
