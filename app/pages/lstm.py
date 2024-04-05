import streamlit as st
import plotly.express as px
import pandas as pd
import pages.helpers as h


st.set_page_config(page_title="Modelo 1")

tab1, tab2 = st.tabs(["Predicciones", "Métricas"])


with tab1:

   data = h.load_data()

   #dropdown_dates  = h.get_dates(data)
   st.header("Predicciones para un mes")
   #selected_date = st.selectbox("Selecciona alguna fecha", dropdown_dates)
   filtered_data = h.filter_data(data )
   print("Logré leer y filtrar")

   
   fig = h.plot_pred("pred_XGB", filtered_data)
   
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