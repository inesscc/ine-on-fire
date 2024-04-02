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

COPY app/app.py /app-dataton/app.py
COPY app/helpers.py /app-dataton/helpers.py


ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
