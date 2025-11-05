FROM python:3.10-slim

WORKDIR /opt/app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y prometheus-node-exporter && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY app.py requirements.txt ./
COPY tests/ ./tests/

RUN pip install --no-cache-dir -r requirements.txt

ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 7860 8000 9100

CMD bash -c "prometheus-node-exporter --web.listen-address=':9100' & python /opt/app/app.py"

