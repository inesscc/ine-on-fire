import plotly.express as px
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

@st.cache_data
def load_data():
    data = pd.read_csv("pages/data/predicciones_test.csv")
    data["ndvi"] = (data["nir08"] - data["red"]) / (data["nir08"] + data["red"])
    return data

def get_dates(data):
    unique_dates = data.time.unique()
    dropdown_dates = unique_dates[:-1]
    return dropdown_dates


def filter_data(data):
    #filtered_data = data[data.time == filter]
    filtered_data = gpd.GeoDataFrame(data,
                                    geometry=gpd.points_from_xy(data.x, data.y),
                                    crs="EPSG:32719"

                                    )
    filtered_data = filtered_data.to_crs("EPSG:4326")

    return filtered_data
    
def plot_pred(pred_col, filtered_data):

      # Crear el mapa utilizando Plotly Express
    fig = go.Figure(data=go.Scattergeo(
         lon=filtered_data.geometry.x,
         lat=filtered_data.geometry.y,
         text=filtered_data["pred_XGB"],
         marker=dict(
            color=filtered_data[pred_col],
            colorscale='Viridis',  # Cambia 'Viridis' al esquema de color que prefieras
            cmin=min(filtered_data[pred_col]),
            cmax=max(filtered_data[pred_col]),
            colorbar=dict(
                  title=pred_col
            )
         ),
         mode='markers'
      ))
    print("Logré hacer el gráfico")

    fig.update_geos(
         center=dict(lon=-71.14894242520268, lat=-33.17148845759217),
         showcountries=True,
         showocean=True,
         showland=True,
         showrivers=True,
         showlakes=True,
         projection_scale=200

      )

    fig.update_layout(
            title = 'Predicciones',
               mapbox_style="outdoors"       

         )
    return fig
