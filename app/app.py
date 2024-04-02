import streamlit as st
import plotly.express as px
import pandas as pd
import helpers as h

tab1, tab2 = st.tabs(["Predicciones", "Métricas"])


with tab1:
   st.header("Predicciones para un mes")
   date = st.selectbox("Selecciona alguna fecha", ["10-03-2023", "17-05-2023", "20-08-2023"])

   # Datos de ejemplo
   data = h.load_and_filter_data()
      

   # Crear el mapa utilizando Plotly Express
   fig = px.scatter_mapbox(data, lat="centroid_lat", lon="centroid_lon", color="peak_hour", size="car_hours",
                            color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                            mapbox_style="carto-positron")
   fig.update_layout(mapbox_style="open-street-map")
   st.plotly_chart(fig, use_container_width=True)


with tab2:
   st.header("Evaluación del modelo")
   data = {
        'Nombre': ['Juan', 'María', 'Carlos'],
        'Edad': [25, 30, 35],
        'Ciudad': ['México', 'Madrid', 'Buenos Aires']
   }

   df = pd.DataFrame(data)

   # Mostrar la tabla
   st.table(df)

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)