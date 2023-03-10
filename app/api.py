
import os
from typing import Union
from pydantic import BaseModel
from fastapi import Depends, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from aiofiles import open as open_aio
from fastapi_versioning import VersionedFastAPI

from .models.LSTM_basic_classifier.model import Model, get_model
from .settings.settings import settings
from .settings.logging_config import LogConfig

# initiate the app and tell it that there is a proxy prefix of /api that gets stripped
# (only effects the loading of the swagger and redoc UIs)
app = FastAPI(title="Transformers API", root_path=settings.docs_ui_root_path)

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
    output_filename = "checkpoint"
    output_file = OUTPUT_DIR + output_filename

    if not new_file:
        return NO_FILE_ERROR_RESPONSE
    else:
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        async with open_aio(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/


@app.post("/lstm-basic-classifier/upload-checkpoint-index/")
async def upload_checkpoint_index(new_file: Union[UploadFile, None] = None):
    output_filename = "my_ckpt.index"
    output_file = OUTPUT_DIR + output_filename

    if not new_file:
        return NO_FILE_ERROR_RESPONSE
    else:
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        async with open_aio(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/


@app.post("/lstm-basic-classifier/upload-checkpoint-data/")
async def upload_checkpoint_data(new_file: Union[UploadFile, None] = None):
    output_filename = "my_ckpt.data-00000-of-00001"
    output_file = OUTPUT_DIR + output_filename

    if not new_file:
        return NO_FILE_ERROR_RESPONSE
    else:
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        async with open_aio(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/


@app.post("/lstm-basic-classifier/upload-train-data/")
async def upload_train_data(new_file: Union[UploadFile, None] = None):
    output_filename = "train_data.csv"
    output_file = OUTPUT_DIR + output_filename

    if not new_file:
        return NO_FILE_ERROR_RESPONSE
    else:
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        async with open_aio(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}


@app.post("/lstm-basic-classifier/refresh-model/")
def refresh_model(model: Model = Depends(get_model)):
    model.refresh_model()
    model.load_weights()

    return {"success": True}


# setup for major versioning
# ensure to copy over all the non-title args to the original FastAPI call...read docs here: https://pypi.org/project/fastapi-versioning/
versioned_app = VersionedFastAPI(app,
                       version_format='{major}',
                       prefix_format='/v{major}', default_api_version=1, root_path=settings.docs_ui_root_path)
