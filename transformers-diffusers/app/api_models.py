from enum import Enum
from pydantic import BaseModel
from typing import Dict, Union, Any, Optional, List


class Framework(Enum):
    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class Task(Enum):
    ASR = "automatic-speech-recognition"


class InferenceRequest(BaseModel):
    inputs: Union[str, List[str], Dict[str, Any], object]
    parameters: Optional[Dict[str, Any]] = None
