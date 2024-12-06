import uvicorn
import logging
import sys
import os
import argparse
import inference_server
from model import Model
from typing import Optional
from instrumentation import setup_instrumentation
import threading
import asyncio


def run_server(port):
    uvicorn.run(inference_server.app, host="0.0.0.0", port=port)


async def load_model(logger, model_id, hf_token):
    try:
        # Load model in background
        model = await Model.create(logger, model_id, hf_token)
        inference_server.model = model
        inference_server.model_status = inference_server.ModelStatus.READY
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        inference_server.model_status = inference_server.ModelStatus.FAILED
        inference_server.model_error = str(e)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run the API server with specified model"
    )
    parser.add_argument("--model-id", type=str, help="The model id to use")
    parser.add_argument("--otlp-endpoint", type=str, help="The OTLP endpoint to use")
    parser.add_argument(
        "--otlp-service-name", type=str, help="The OTLP service name to use"
    )
    args = parser.parse_args()

    MODEL_ID: Optional[str] = args.model_id or os.getenv("MODEL_ID")
    if not MODEL_ID:
        raise ValueError(
            "MODEL_ID must be defined in environment variables or arguments"
        )
    HF_API_TOKEN: Optional[str] = os.getenv("HF_API_TOKEN")
    PORT = int(os.getenv("PORT", 80))

    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") and args.otlp_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = args.otlp_endpoint

    if not os.getenv("OTLP_SERVICE_NAME") and args.otlp_service_name:
        os.environ["OTLP_SERVICE_NAME"] = args.otlp_service_name

    # Setup logging
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    setup_instrumentation(logger)

    # Start the server in a separate thread
    server_thread = threading.Thread(target=run_server, args=(PORT,))
    server_thread.start()

    # Run model loading in the main thread
    asyncio.run(load_model(logger, MODEL_ID, HF_API_TOKEN))


if __name__ == "__main__":
    main()
