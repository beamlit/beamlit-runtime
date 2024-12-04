from huggingface_hub import configure_http_backend
import uvicorn
import logging
import sys
import os
import argparse
import inference_server
from dragonfly import set_dragonfly_url, backend_factory
from model import Model
from typing import Optional


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the API server with specified model"
    )
    parser.add_argument("--model-id", type=str, help="The model id to use")
    args = parser.parse_args()

    MODEL_ID: Optional[str] = args.model_id or os.getenv("MODEL_ID")
    if not MODEL_ID:
        raise ValueError(
            "MODEL_ID must be defined in environment variables or arguments"
        )
    HF_API_TOKEN: Optional[str] = os.getenv("HF_API_TOKEN")
    DRAGONFLY_URL: Optional[str] = os.getenv("DRAGONFLY_URL")
    DEACTIVATE_DRAGONFLY: Optional[bool] = os.getenv("DEACTIVATE_DRAGONFLY")

    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    if DRAGONFLY_URL:
        set_dragonfly_url(DRAGONFLY_URL)
    if DEACTIVATE_DRAGONFLY is None:
        configure_http_backend(backend_factory=backend_factory)

    try:
        inference_server.model = Model(logger, MODEL_ID, HF_API_TOKEN)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    uvicorn.run(inference_server.app, host="0.0.0.0", port=80)
