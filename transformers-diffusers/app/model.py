import logging
import requests
from typing import Dict, Any, Optional, Union
from diffusers import DiffusionPipeline
from transformers import pipeline as transformers_pipeline, Pipeline
import torch
from api_models import Framework
from typing import Optional


class Model:

    pipeline: Optional[Union[DiffusionPipeline, Pipeline]] = None
    framework: Optional[Framework] = None
    model_id: Optional[str] = None
    hf_api_token: Optional[str] = None

    def __init__(
        self, logger: logging.Logger, model_id: str, hf_api_token: Optional[str]
    ):
        self.logger = logger
        self.model_id = model_id
        self.hf_api_token = hf_api_token

        self.logger.info(f"Loading model {self.model_id}")

        # Fetch model metadata
        metadata = self.get_model_metadata()
        self.logger.info("Fetched model metadata (20%)")

        tags = metadata.get("tags", [])

        # Determine framework from tags
        if "transformers" in tags:
            self.framework = Framework.TRANSFORMERS
        elif "diffusers" in tags:
            self.framework = Framework.DIFFUSERS
        else:
            raise ValueError(f"Framework not found in model tags for {self.model_id}")

        self.logger.info(f"Determined framework: {self.framework} (40%)")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.framework == Framework.TRANSFORMERS:
            try:
                optional_args = {}
                if self.hf_api_token:
                    optional_args["use_auth_token"] = self.hf_api_token
                self.pipeline = transformers_pipeline(
                    model=self.model_id,
                    device_map=device,
                    **optional_args,
                )
                self.logger.info(
                    f"Model {self.model_id} loaded successfully using {self.framework} framework (100%)"
                )
            except Exception as e:
                self.logger.error(f"Error loading transformers model: {e}")
                raise
        elif self.framework == Framework.DIFFUSERS:
            try:
                optional_args = {}
                if self.hf_api_token:
                    optional_args["use_auth_token"] = self.hf_api_token
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    device_map=device,
                    **optional_args,
                )
                self.logger.info(
                    f"Model {self.model_id} loaded successfully using {self.framework} framework (100%)"
                )
            except Exception as e:
                self.logger.error(f"Error loading diffusers model: {e}")
                raise
        else:
            raise ValueError(f"Invalid framework: {self.framework}")

    def get_model_metadata(self) -> Dict[str, Any]:
        url = f"https://huggingface.co/api/models/{self.model_id}"
        headers = {}
        if self.hf_api_token:
            headers["Authorization"] = f"Bearer {self.hf_api_token}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch model metadata for {self.model_id}")
        return response.json()
