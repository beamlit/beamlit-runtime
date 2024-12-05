import uvicorn
import logging
import sys
import os
import argparse
import inference_server
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
    PORT = os.getenv("PORT", 80)

    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    try:
        inference_server.model = Model(logger, MODEL_ID, HF_API_TOKEN)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    uvicorn.run(inference_server.app, host="0.0.0.0", port=int(PORT))
