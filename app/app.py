import streamlit as st
import plotly.express as px
import pandas as pd
import helpers as h
import plotly.graph_objects as go



tab1, tab2 = st.tabs(["Predicciones", "Métricas"])


with tab1:

   data = h.load_data()

   dropdown_dates  = h.get_dates(data)
   st.header("Predicciones para un mes")
   selected_date = st.selectbox("Selecciona alguna fecha", dropdown_dates)
   filtered_data = h.filter_data(data, selected_date )
   print("Logré leer y filtrar")

   # Crear el mapa utilizando Plotly Express
   fig = go.Figure(data=go.Scattergeo(
      lon=filtered_data.geometry.x,
      lat=filtered_data.geometry.y,
      marker=dict(
         color=filtered_data["ndvi"],
         colorscale='Viridis',  # Cambia 'Viridis' al esquema de color que prefieras
         cmin=min(filtered_data["ndvi"]),
         cmax=max(filtered_data["ndvi"]),
         colorbar=dict(
               title="NDVI"
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

   print("Logré hacer todo el gráfico")

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