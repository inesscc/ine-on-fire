import plotly.express as px
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    #data["ndvi"] = (data["nir08"] - data["red"]) / (data["nir08"] + data["red"])
    return data

def get_dates(data):
    unique_dates = data.time.unique()
    dropdown_dates = unique_dates
    return dropdown_dates

def get_dates_2(data):
    unique_dates = data.fecha_ini.unique()
    dropdown_dates = unique_dates
    return dropdown_dates


def filter_data(data, filter):
    filtered_data = data[data.time == filter]
    filtered_data = gpd.GeoDataFrame(filtered_data,
                                    geometry=gpd.points_from_xy(filtered_data.x, filtered_data.y),
                                    crs="EPSG:32719"

                                    )
    filtered_data = filtered_data.to_crs("EPSG:4326")

    return filtered_data

def filter_data_2(data, filter):
    filtered_data = data[data.fecha_ini == filter]
    filtered_data = gpd.GeoDataFrame(filtered_data,
                                    geometry=gpd.points_from_xy(filtered_data.x, filtered_data.y),
                                    crs="EPSG:32719"

                                    )
    filtered_data = filtered_data.to_crs("EPSG:4326")

    return filtered_data
    
def plot_pred(pred_col, filtered_data):

      # Crear el mapa utilizando Plotly Express
    fig = go.Figure(data=go.Scattergeo(
         lon=filtered_data.geometry.x,
         lat=filtered_data.geometry.y,
         text=filtered_data[pred_col],
         marker=dict(
            color=filtered_data[pred_col],
            colorscale='Viridis',  # Cambia 'Viridis' al esquema de color que prefieras
            # cmin=min(filtered_data[pred_col]),
            cmin=0.1,            
            # cmax=max(filtered_data[pred_col]),
            cmax=0.8,            
            colorbar=dict(
                #   title=pred_col
                  title="pred_prob"
            )
         ),
         mode='markers'
      ))
    print("Logré hacer el gráfico")

    fig.update_geos(
         center=dict(lon=-71.14894242520268, lat=-33.17148845759217),
         showcountries=True,
         showocean=True,
         #showland=True,
         showrivers=True,
         showlakes=True,
         projection_scale=1

      )

    fig.update_layout(
            title = 'Predicciones',
            mapbox_style="outdoors",
            height=800,  # Ajustar la altura del gráfico
            width=500    # Ajustar el ancho del gráfico       

         )
    
    fig.update_layout(
    geo=dict(
        resolution=110,
        lonaxis=dict(range=[-75, -70]),
        lataxis=dict(range=[-34, -30]),
    )
    ) 

    return fig
