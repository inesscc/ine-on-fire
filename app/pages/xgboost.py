import streamlit as st
import plotly.express as px
import pandas as pd
import pages.helpers as h


st.set_page_config(page_title="Modelo 2")

tab1, tab2 = st.tabs(["Predicciones", "Métricas"])


with tab1:

   data = h.load_data("pages/data/predicciones_test_xgb.csv")

   dropdown_dates  = h.get_dates(data)
   st.header("Predicciones para un día")
   selected_date = st.selectbox("Selecciona alguna fecha", dropdown_dates)
   filtered_data = h.filter_data(data, selected_date )
   print(selected_date)

   
   fig = h.plot_pred("pred_XGB", filtered_data)
   
   print("Logré hacer todo el gráfico")

   st.plotly_chart(fig, use_container_width=True)


with tab2:
   
   metricas = h.load_data("pages/data/metricas_xgb.csv")

   st.header("Evaluación del modelo")
   

   df = pd.DataFrame(metricas)

   # Mostrar la tabla
   st.table(df)

   st.write("### Variables más importantes para el modelo")
 
   st.image('pages/images/variables_importantes_xgb.png')

   st.write("### Matriz de confusión")
   st.image('pages/images/matriz_confusion_xgb.png')

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)