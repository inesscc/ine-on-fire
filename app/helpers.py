import plotly.express as px
import streamlit as st


@st.cache_data
def load_and_filter_data():
    data = px.data.carshare()
    return data