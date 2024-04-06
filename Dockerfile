FROM python:3.10.8-slim

RUN mkdir app-dataton

WORKDIR /app-dataton

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app-dataton/requirements.txt
RUN pip3 install -r requirements.txt

# Copiamos solo los archivos que contienen código
COPY app/ine-on-fire.py /app-dataton/ine-on-fire.py
COPY app/pages/helpers.py /app-dataton/pages/helpers.py
COPY app/pages/lstm.py /app-dataton/pages/lstm.py
COPY app/pages/xgboost.py /app-dataton/pages/xgboost.py


ENTRYPOINT ["streamlit", "run", "ine-on-fire.py", "--server.port=8501", "--server.address=0.0.0.0"]
