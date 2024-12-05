FROM python:3.12

RUN pip install diffusers "transformers[torch]" fastapi uvicorn sentence-transformers accelerate>=0.26.0
RUN apt-get update && apt-get install -y ffmpeg

COPY app /app

ENTRYPOINT ["python", "app/main.py"]
