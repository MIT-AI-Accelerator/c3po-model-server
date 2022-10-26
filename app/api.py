from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from logging.config import dictConfig
import logging
from app.loggingConfig import LogConfig

from app.models.GPT_J_seq2seq.model import Model
from app.models.GPT_J_seq2seq.model import get_model

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
    answer = model.predict(request.text)
    logger.debug("start--" + request.text + "--end")
    return PromptResponse(
        answer = answer
    )
