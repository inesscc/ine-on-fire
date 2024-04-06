# Readme INE on Fire

Este documento pretende explicar cómo correr el modelo de INE on Fire y la estructura básica de sus scripts. 

## Extracción de datos

En primera instancia, se extraen datos desde el datacube en el notebook [datacube/extraer_datos.ipynb](https://github.com/inesscc/ine-on-fire/blob/main/datacube/extraer_datos.ipynb). Como es de esperar, este notebook corre exclusivamente desde el datacube y genera insumos para el modelo. Este se encarga de las siguientes acciones:

1. Utiliza las bases de incendios de CONAF para obtener un set de 3000 píxeles aleatorios del bounding box de cada uno de los incendios que existen en el período de recolección de SENTINEL-2 para Valparaíso (total: 24 incendios). Se recolectan todas las imágenes hasta 25 días antes del incendio y hasta 10 días después del incendio. Luego, se eliminan imágenes que vengan con un porcentaje de píxeles inválidos mayor al 25%. Los píxeles dentro del polígono del incendio de se marcan con un 1 en la variable es_incendio y los píxeles fuera, se marcan con un 0.
2. Se obtiene el bounding box de valparaíso y se recolectan imágenes para las mismas fechas capturadas en los otros 24 incendios. Se recolectan 9000 píxeles aleatorios para el bounding box de Valparaíso y luego se mantienen solo los que están dentro del polígono de Valparaíso en sí. 
3. Los datos se consolidan y se exportan.
4. Se generan datos de elevación para Valparaíso y se exportan (no fueron utilizados finalmente).
5. Se obtienen datos de incendio en Valparaíso de SAF. Se determina que la imagen de 2024-02-29 contiene un incendio que también puede ser detectado con SENTINEL-2. Debido a que las imágenes SAF no contienen las bandas que nuestro modelo utiliza, se opta por utilizar la imagen como referencia para testear sobre imágenes de SENTINEL-2 los resultados del modelo.
   1. Se exportan las imágenes de SAF y SENTINEL-2, antes y después del incendio, en el segundo caso.

Los datos exportados se encuentran aquí: 

[datos_consolidados_sentinel2_v4.csv](https://drive.google.com/file/d/1kvHDOSgTqMMwqzLe7yieMH0UjQFKxg_e/view?usp=drive_link)

## Consolidación de datos

Luego, los datos exportados son complementados con varias otros datos provienen de distintas fuentes:

- Sentinel2: se utilizaron los índices NVDI, EVI y NDWI, obtenidos de Cubo de Datos

- Google Earth Engine: datos de clima, provenientes del dataset ERA5

- Biblioteca del Congreso Nacional: Se calcularan distancias de cada pixel a los siguientes puntos: fuente de agua, zona urbana, red vial, áreas silvestres.

Esta consolidación se hace en el notebook [notebooks/estimacion_dist_gee.ipynb](https://github.com/inesscc/ine-on-fire/blob/main/notebooks/estimacion_dist_gee.ipynb) y los datos consolidados de esta forma están aquí:

[data_consolidados_sentinel2_distancias_indices_era5](https://drive.google.com/file/d/1wZPiMqg6tzRgqy62DRVIEYL8vkFPBsSb/view?usp=drive_link)


## Entrenamiento de modelos

Finalmente, usados los datos consolidados para entrenar dos tipos modelos:

LSTM: Recibe como entrada las últimas 3 capturas de un pixel antes de un incendio, a lo que se agregan datos de distancia y clima.

- Para reproducir el entrenamiento del LSTM, consultar el archivo [model/README.md](https://github.com/inesscc/ine-on-fire/blob/main/model/README.md)

XGBoost: Recibe como entrada los datos de la captura inmediatamente anterior al incendio y un promedio de los momentos en t-1, t-2 y t-3. Adicionalmente, se incluyen datos de distancia y clima.

- Para reproducir el entrenamiento del modelo XGBoost, consultar el notebook [notebooks/modelo xgb/ModXGB_sin_tiempo.ipynb](https://github.com/inesscc/ine-on-fire/blob/main/notebooks/modelo%20xgb/ModXGB_sin_tiempo.ipynb)

## Presentación

https://inesscc.github.io/ine-on-fire/#/section

## Aplicación Demo

http://64.23.252.231:8010/
