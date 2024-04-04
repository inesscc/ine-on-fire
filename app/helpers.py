import plotly.express as px
import streamlit as st
import pandas as pd
import geopandas as gpd

@st.cache_data
def load_data():
    data = pd.read_csv("data/data_con_dependiente.csv")
    data["ndvi"] = (data["nir08"] - data["red"]) / (data["nir08"] + data["red"])
    return data

def get_dates(data):
    unique_dates = data.time.unique()
    dropdown_dates = unique_dates[:-1]
    return dropdown_dates


def filter_data(data, filter):
    filtered_data = data[data.time == filter]
    filtered_data = gpd.GeoDataFrame(filtered_data,
                                    geometry=gpd.points_from_xy(filtered_data.x, filtered_data.y),
                                    crs="EPSG:32719"

                                    )
    filtered_data = filtered_data.to_crs("EPSG:4326")

    return filtered_data
    
    