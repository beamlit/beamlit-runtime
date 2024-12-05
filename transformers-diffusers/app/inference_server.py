from fastapi import FastAPI
import traceback
from typing import Optional, Any
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from api_models import InferenceRequest, Task
import logging
from model import Model
import base64

logger = logging.getLogger(__name__)

app = FastAPI(
    title="inference_server",
    description="Inference server for transformers and diffusers models",
)

model: Optional[Model] = None


@app.get("/health")
def health():
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/")
def infer(request: InferenceRequest):
    logger.info("Received request")

    if not _is_model_ready():
        return JSONResponse(
            status_code=400, content={"error message": "Model not loaded"}
        )

    request.parameters = request.parameters or {}

    try:
        resp = _process_request(request)
        return JSONResponse(status_code=200, content=resp)
    except Exception:
        logger.error("Exception occurred: %s", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error message": "An internal error has occurred."},
        )


def _is_model_ready() -> bool:
    return model is not None and model.pipeline is not None


def _process_request(request: InferenceRequest) -> Any:
    if not model or not model.pipeline:
        raise ValueError("Model not initialized")

    if isinstance(request.inputs, str):
        return _handle_string_input(request)
    elif isinstance(request.inputs, list):
        return _handle_list_input(request)
    elif isinstance(request.inputs, dict):
        return _handle_dict_input(request)
    raise ValueError(f"Unsupported input type: {type(request.inputs)}")


def _handle_string_input(request: InferenceRequest) -> Any:
    if not model or not model.pipeline:
        raise ValueError("Model not initialized")

    if _is_asr_with_base64(request):
        input_bytes = base64.b64decode(str(request.inputs))
        return model.pipeline(input_bytes, **(request.parameters or {}))

    if isinstance(request.inputs, str):
        if isinstance(model.pipeline, SentenceTransformer):
            embeddings = model.pipeline.encode([str(request.inputs)])
            return embeddings[0].tolist()
        return model.pipeline(str(request.inputs), **(request.parameters or {}))


def _handle_list_input(request: InferenceRequest) -> Any:
    if not model or not model.pipeline:
        raise ValueError("Model not initialized")
    if isinstance(request.inputs, list):
        if isinstance(model.pipeline, SentenceTransformer):
            embeddings = model.pipeline.encode([str(i) for i in request.inputs])
            return [e.tolist() for e in embeddings]
        return model.pipeline(*request.inputs, **(request.parameters or {}))


def _handle_dict_input(request: InferenceRequest) -> Any:
    if not model or not model.pipeline:
        raise ValueError("Model not initialized")
    if isinstance(request.inputs, dict):
        return model.pipeline(**request.inputs, **(request.parameters or {}))


def _is_asr_with_base64(request: InferenceRequest) -> bool:
    if not model:
        return False
    return (
        model.task == Task.ASR
        and isinstance(request.inputs, str)
        and not str(request.inputs).startswith(("http", "https"))
    )
