from fastapi import FastAPI
import traceback
from fastapi.responses import JSONResponse
from api_models import InferenceRequest, Task
import logging
from model import Model
from typing import Optional
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

    if model is None or model.pipeline is None:
      return JSONResponse(status_code=400, content={"error message": "Model not loaded"})
    if request.parameters is None:
        request.parameters = {}
    resp = None

    if isinstance(request.inputs, str):
        try:
            if model.task == Task.ASR and not (
                str(request.inputs).startswith("http")
                or str(request.inputs).startswith("https")
            ):
                resp = model.pipeline(
                    base64.b64decode(request.inputs), **request.parameters
                )
            else:
                resp = model.pipeline(request.inputs, **request.parameters)
        except Exception as e:
            logger.error("Exception occurred: %s", traceback.format_exc())
            return JSONResponse(status_code=500, content={"error message": "An internal error has occurred."})
    elif isinstance(request.inputs, list):
        try:
            resp = model.pipeline(*request.inputs, **request.parameters)
        except Exception as e:
            logger.error("Exception occurred: %s", traceback.format_exc())
            return JSONResponse(status_code=500, content={"error message": "An internal error has occurred."})
    elif isinstance(request.inputs, dict):
        try:
            resp = model.pipeline(**request.inputs, **request.parameters)
        except Exception as e:
            logger.error("Exception occurred: %s", traceback.format_exc())
            return JSONResponse(status_code=500, content={"error message": "An internal error has occurred."})
    return JSONResponse(status_code=200, content=resp)
