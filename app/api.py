
from pydantic import BaseModel
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Union

from app.models.LSTM_basic_classifier.model import Model, get_model

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


@app.post("/predict/", response_model=PromptResponse)
def predict(request: PromptRequest, model: Model = Depends(get_model)):
    answer = model.classify_single_label(request.text)

    return PromptResponse(
        answer=answer
    )

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/
@app.post("/uploadfile/")
async def create_upload_file(file: Union[UploadFile, None] = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        return {"filename": file.filename}
