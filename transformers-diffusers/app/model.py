import logging
import requests
from typing import Dict, Any, Optional, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import pipeline as transformers_pipeline, Pipeline
from sentence_transformers import SentenceTransformer
import torch
from api_models import Framework, Task


class Model:
    pipeline: Optional[Union[Any, Pipeline, SentenceTransformer]] = None
    framework: Optional[Framework] = None
    task: Optional[Task] = None
    model_id: Optional[str] = None
    hf_api_token: Optional[str] = None

    def __init__(
        self, logger: logging.Logger, model_id: str, hf_api_token: Optional[str]
    ):
        self.logger = logger
        self.model_id = model_id
        self.hf_api_token = hf_api_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Loading model {self.model_id}")

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model by fetching metadata and loading the appropriate pipeline"""
        metadata = self.get_model_metadata()
        self.logger.info("Fetched model metadata (20%)")

        self._set_task(metadata)
        self._set_framework(metadata)
        self._load_pipeline()

    def _set_task(self, metadata: Dict[str, Any]):
        """Set the task based on metadata"""
        try:
            self.task = Task(metadata.get("pipeline_tag", None))
        except ValueError as e:
            self.logger.info(f"Error determining task: {e}, support might be limited")
            self.task = None

    def _set_framework(self, metadata: Dict[str, Any]):
        """Determine and set the framework based on model tags"""
        tags = metadata.get("tags", [])

        framework_mapping = {
            "sentence-transformers": Framework.SENTENCE_TRANSFORMERS,
            "transformers": Framework.TRANSFORMERS,
            "diffusers": Framework.DIFFUSERS,
        }

        for tag, framework in framework_mapping.items():
            if tag in tags:
                self.framework = framework
                break
        else:
            raise ValueError(f"Framework not found in model tags for {self.model_id}")

        self.logger.info(f"Determined framework: {self.framework} (40%)")

    def _load_pipeline(self):
        """Load the appropriate pipeline based on the framework"""
        loader_map = {
            Framework.TRANSFORMERS: self._load_transformers_pipeline,
            Framework.DIFFUSERS: self._load_diffusers_pipeline,
            Framework.SENTENCE_TRANSFORMERS: self._load_sentence_transformers_pipeline,
        }

        loader = loader_map.get(self.framework) if self.framework is not None else None
        if not loader:
            raise ValueError(f"Invalid framework: {self.framework}")

        loader()

    def _load_transformers_pipeline(self):
        """Load a transformers pipeline"""
        try:
            optional_args: Dict[str, Any] = (
                {"use_auth_token": self.hf_api_token} if self.hf_api_token else {}
            )
            self.pipeline = transformers_pipeline(
                model=self.model_id,
                device_map=self.device,
                **optional_args,
            )
            self._log_success()
        except Exception as e:
            self._log_error("transformers", e)

    def _load_diffusers_pipeline(self):
        """Load a diffusers pipeline"""
        try:
            optional_args: Dict[str, Any] = (
                {"use_auth_token": self.hf_api_token} if self.hf_api_token else {}
            )
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                **optional_args,
            )
            self._log_success()
        except Exception as e:
            self._log_error("diffusers", e)

    def _load_sentence_transformers_pipeline(self):
        """Load a sentence-transformers pipeline"""
        try:
            self.pipeline = SentenceTransformer(self.model_id)
            self._log_success()
        except Exception as e:
            self._log_error("sentence-transformers", e)

    def _log_success(self):
        """Log successful model loading"""
        self.logger.info(
            f"Model {self.model_id} loaded successfully using {self.framework} framework (100%)"
        )

    def _log_error(self, framework_name: str, error: Exception):
        """Log error and raise exception"""
        self.logger.error(f"Error loading {framework_name} model: {error}")
        raise error

    def get_model_metadata(self) -> Dict[str, Any]:
        """Fetch model metadata from Hugging Face API"""
        url = f"https://huggingface.co/api/models/{self.model_id}"
        headers = {}
        if self.hf_api_token:
            headers["Authorization"] = f"Bearer {self.hf_api_token}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch model metadata for {self.model_id}")
        return response.json()
