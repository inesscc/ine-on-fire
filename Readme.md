# Readme INE on Fire

Este documento pretende explicar cómo correr el modelo de INE on Fire y la estructura básica de sus scripts. 

En primera instancia, se extraen datos desde el datacube en el notebook `datacube/extraer_datos.ipynb`. Como es de esperar, este notebook corre exclusivamente desde el datacube y genera insumos para el modelo. Este se encarga de las siguientes acciones:

1. Utiliza las bases de incendios de CONAF para obtener un set de 3000 píxeles aleatorios del bounding box de cada uno de los incendios que existen en el período de recolección de SENTINEL-2 para Valparaíso (total: 24 incendios). Se recolectan todas las imágenes hasta 25 días antes del incendio y hasta 10 días después del incendio. Luego, se eliminan imágenes que vengan con un porcentaje de píxeles inválidos mayor al 25%. Los píxeles dentro del polígono del incendio de se marcan con un 1 en la variable es_incendio y los píxeles fuera, se marcan con un 0.
2. Se obtiene el bounding box de valparaíso y se recolectan imágenes para las mismas fechas capturadas en los otros 24 incendios. Se recolectan 9000 píxeles aleatorios para el bounding box de Valparaíso y luego se mantienen solo los que están dentro del polígono de Valparaíso en sí. 
3. Los datos se consolidan y se exportan.
4. Se generan datos de elevación para Valparaíso y se exportan (no fueron utilizados finalmente).
5. Se obtienen datos de incendio en Valparaíso de SAF. Se determina que la imagen de 2024-02-29 contiene un incendio que también puede ser detectado con SENTINEL-2. Debido a que las imágenes SAF no contienen las bandas que nuestro modelo utiliza, se opta por utilizar la imagen como referencia para testear sobre imágenes de SENTINEL-2 los resultados del modelo.
   1. Se exportan las imágenes de SAF y SENTINEL-2, antes y después del incendio, en el segundo caso.

Luego de exportar los datos...