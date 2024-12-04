from fastapi import FastAPI
from fastapi.responses import JSONResponse
from api_models import InferenceRequest
import logging
from model import Model
from typing import Optional

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
    logger.info("Received request", extra={"request": request})
    if request.parameters is None:
        request.parameters = {}
    if isinstance(request.inputs, str):
        resp = model.pipeline(request.inputs, **request.parameters)
    elif isinstance(request.inputs, list):
        resp = model.pipeline(*request.inputs, **request.parameters)
    else:
        resp = model.pipeline(**request.inputs, **request.parameters)
    return JSONResponse(status_code=200, content=resp)
