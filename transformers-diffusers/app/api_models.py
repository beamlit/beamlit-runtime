from enum import Enum
from pydantic import BaseModel
from typing import Dict, Union, Any, Optional, List


class Framework(Enum):
    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"


class InferenceRequest(BaseModel):
    inputs: Union[str, List[str], object]
    parameters: Optional[Dict[str, Any]] = None
