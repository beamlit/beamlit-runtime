FROM python:3.12

RUN pip install diffusers "transformers[torch]" fastapi prometheus_client opentelemetry-exporter-otlp uvicorn sentence-transformers opentelemetry-instrumentation-fastapi opentelemetry-sdk accelerate>=0.26.0
RUN apt-get update && apt-get install -y ffmpeg

COPY app /app

ENTRYPOINT ["python", "app/main.py"]
