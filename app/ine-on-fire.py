import streamlit as st
from st_pages import Page, show_pages


st.set_page_config(
    page_title="INE ON FIRE")


show_pages(
    [
        Page("ine-on-fire.py"),
        Page("pages/lstm.py"),
        Page("pages/xgboost.py")

    ]
)

st.write("# INE ON FIRE 👋")

#st.sidebar.success("Select a demo above.")

st.image('pages/images/app_code.png', caption = '')


st.markdown(
"""
### Esta aplicación muestra resultados de 2 modelos entrenados para la predicción de incendios en Valparaíso. 

#### Los datos utilizados en el entrenamiento provienen de distintas fuentes:

- Sentinel2: se utilizaron los índices NVDI, EVI y NDWI, obtenidos de Cubo de Datos
- Google Earth Engine: datos de clima, provenientes del dataset ERA5   
- Biblioteca del Congreso Nacional: Se calcularan distancias de cada pixel a los siguientes puntos: fuente de agua, zona urbana, red vial, áreas silvestres.
- CONAF: polígonos de incendios. Estos datos se utilizaron para el etiquetado

#### Los modelos testeados son:

- **LSTM**: Recibe como entrada las últimas 4 capturas de un pixel antes de un incendio, a lo que se agregan datos de distancia y clima.   
- **xgboost**: Recibe como entrada los datos de la captura inmediatamente anterior al incendio y un promedio de los momentos en t-1, t-2 y t-3. Adicionalmente, 
se incluyen datos de distancia y clima.    


"""
)