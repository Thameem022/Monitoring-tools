# 🌱 Sustainable.ai - MLOps Monitoring

A sustainability chatbot with Prometheus metrics monitoring.

## Quick Start

1. Create `.env` file:
   ```bash
   echo "HF_TOKEN=your_token_here" > .env
   ```

2. Start all services:
   ```bash
   docker-compose up -d
   ```

3. Access services:
   - **Gradio App**: http://localhost:7860
   - **Prometheus**: http://localhost:9090
   - **Grafana**: http://localhost:3000 (admin/admin)
   - **Metrics**: http://localhost:8000/metrics

## Metrics

- `app_model_usage_total` - Model invocations by type (local/remote)
- `app_carbon_footprint_kg` - Calculated carbon footprints
- `app_streaming_chunks_total` - Streaming chunks received
- `app_message_length_chars` - User message lengths
- `app_inference_time_seconds` - Model inference time
- `app_active_users_current` - Current active users

## Stop Services

```bash
docker-compose down
```
