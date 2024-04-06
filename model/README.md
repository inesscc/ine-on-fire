[![DOI](https://zenodo.org/badge/489363446.svg)](https://zenodo.org/badge/latestdoi/489363446)


# Intro 

Este directorio contiene nuestra adaptación del código publicado junto al siguiente artículo académico:

[Wildfire Danger Prediction and Understanding with Deep Learning](https://doi.org/10.1029/2022gl099368), published in [Geophysical Research Letters (https://doi.org/10.1029/2022GL099368)](https://agupubs.onlinelibrary.wiley.com/journal/19448007).

Authored by Spyros Kondylatos, Ioannis Prapas, Michele Ronco, Ioannis Papoutsis, Gustau Camps-Valls, Maria Piles, Miguel-Angel Fernandez-Torres, Nuno Carvalhais

# Entrenamiento del modelo LSTM

Se puede entrenar el modelo LSTM que arroja las predicciones y métricas presentadas en esta DO-safíos Datatón, con el siguiente comando:

`python run.py experiment=lstm_temporal_cls`

Los datos usados para el entrenamiento del modelo pueden descargarse desde este drive:

https://drive.google.com/drive/folders/19AnBchYbSOOCJqEOaF9a8V6OXudG7_ul?usp=drive_link

Los detalles del flujo de entrenamiento, validación y testeo pueden consultarse en el repositorio original:

https://github.com/Orion-AI-Lab/wildfire_forecasting
