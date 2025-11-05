---
title: Case Study 3 - Monitoring Tools
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
short_description: Sustainable.ai - Carbon Footprint Chatbot with Prometheus Monitoring
---

# 🌱 Sustainable.ai - Monitoring Tools

A sustainability-focused AI chatbot that helps users understand and reduce their carbon footprint. The application features comprehensive Prometheus metrics for monitoring and observability.

## Features

- **Carbon Footprint Calculator**: Calculate your weekly carbon footprint based on transportation and food choices
- **AI-Powered Sustainability Advisor**: Get personalized suggestions to reduce your environmental impact
- **Dual Model Support**: Choose between local model (google/gemma-3-270m-it) or remote Hugging Face API (gpt-oss-20b)
- **Prometheus Metrics**: Built-in monitoring with multiple custom metrics
- **System Metrics**: Node exporter for system-level monitoring

## Prometheus Metrics

The application exposes the following custom metrics on port `8000`:

| Metric | Type | Description |
|--------|------|-------------|
| `app_model_usage_total` | Counter | Total number of model invocations (labeled by `model_type`: `local` or `remote`) |
| `app_carbon_footprint_kg` | Summary | Calculated carbon footprint in kg CO2 |
| `app_streaming_chunks_total` | Counter | Total number of streaming chunks received from remote API |
| `app_message_length_chars` | Histogram | Length of user messages in characters (buckets: 10, 50, 100, 200, 500, 1000) |
| `app_inference_time_seconds` | Summary | Time spent in model inference |
| `app_active_users_current` | Gauge | Current number of active users |

**Metrics Endpoint**: `http://localhost:8000/metrics`

## Ports

- **7860**: Gradio web UI
- **8000**: Prometheus client metrics endpoint
- **9100**: Prometheus node-exporter system metrics

## Setup

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- Hugging Face token (`HF_TOKEN`)

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Hugging Face token:
```bash
export HF_TOKEN=your_token_here
```

3. Run the application:
```bash
python app.py
```

4. Access the application:
   - Web UI: http://localhost:7860
   - Metrics: http://localhost:8000/metrics
   - System Metrics: http://localhost:9100/metrics

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t monitoring-tools .
```

2. Run the container:
```bash
docker run -p 7860:7860 -p 8000:8000 -p 9100:9100 -e HF_TOKEN=your_token_here monitoring-tools
```

The container will automatically:
- Start Prometheus node-exporter on port 9100
- Expose application metrics on port 8000
- Launch the Gradio web UI on port 7860
