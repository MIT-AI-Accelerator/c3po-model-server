
from pydantic import BaseModel
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from aiofiles import open
import os

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

class ErrorResponse(BaseModel):
    message: str

NO_FILE_ERROR_RESPONSE = ErrorResponse(message="No upload file sent")
OUTPUT_DIR = "./app/models/LSTM_basic_classifier/training_checkpoints/"

@app.post("/predict/", response_model=PromptResponse)
def predict(request: PromptRequest, model: Model = Depends(get_model)):
    answer = model.classify_single_label(request.text)

    return PromptResponse(
        answer=answer
    )

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/
@app.post("/lstm-basic-classifier/upload-checkpoint-metadata/")
async def upload_checkpoint_metadata(new_file: Union[UploadFile, None] = None):
    output_dir = "./app/models/LSTM_basic_classifier/training_checkpoints/"
    output_filename = "checkpoint"
    output_file = output_dir + output_filename

    if not new_file:
        return {"message": "No upload file sent"}
    else:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        async with open(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/
@app.post("/lstm-basic-classifier/upload-checkpoint-index/")
async def upload_checkpoint_index(new_file: Union[UploadFile, None] = None):
    output_dir = "./app/models/LSTM_basic_classifier/training_checkpoints/"
    output_filename = "my_ckpt.index"
    output_file = output_dir + output_filename

    if not new_file:
        return {"message": "No upload file sent"}
    else:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        async with open(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/
@app.post("/lstm-basic-classifier/upload-checkpoint-data/")
async def upload_checkpoint_data(new_file: Union[UploadFile, None] = None):
    output_dir = "./app/models/LSTM_basic_classifier/training_checkpoints/"
    output_filename = "my_ckpt.data-00000-of-00001"
    output_file = output_dir + output_filename

    if not new_file:
        return {"message": "No upload file sent"}
    else:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        async with open(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/
@app.post("/lstm-basic-classifier/upload-train-data/")
async def upload_train_data(new_file: Union[UploadFile, None] = None):
    output_dir = "./app/models/LSTM_basic_classifier/training_checkpoints/"
    output_filename = "train_data.csv"
    output_file = output_dir + output_filename

    if not new_file:
        return {"message": "No upload file sent"}
    else:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        async with open(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

@app.post("/lstm-basic-classifier/refresh-model/")
def refresh_model(model: Model = Depends(get_model)):
    model.refresh_model()
    model.load_weights()

    return {"success": True}
