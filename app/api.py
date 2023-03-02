from logging.config import dictConfig
import logging
from pydantic import BaseModel
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.logging_config import LogConfig

from app.models.LSTM_basic_classifier.model import Model
from app.models.LSTM_basic_classifier.model import get_model

dictConfig(LogConfig().dict())
logger = logging.getLogger("mycoolapp")

logger.info("Dummy Info")
logger.error("Dummy Error")
logger.debug("Dummy Debug")
logger.warning("Dummy Warning")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    text: str


class PromptResponse(BaseModel):
    answer: str


@app.post("/predict", response_model=PromptResponse)
def predict(request: PromptRequest, model: Model = Depends(get_model)):
    answer = model.classify_single_label(request.text)
    logger.debug("start--%s--end",request.text)
    return PromptResponse(
        answer = answer
    )
