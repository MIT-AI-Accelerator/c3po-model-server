import os
from typing import Union
from pydantic import BaseModel
from fastapi import Depends, APIRouter, UploadFile
from aiofiles import open as open_aio

from app.dependencies import get_lstm_stress_classifier_model
from ..ai_service.inference.inference_model import LstmStressClassifierModel, BASE_CKPT_DIR

router = APIRouter(
    prefix=""
)

class ErrorResponse(BaseModel):
    message: str

NO_FILE_ERROR_RESPONSE = ErrorResponse(message="No upload file sent")

# os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data_open/example_data.csv")

# upload file docs here: https://fastapi.tiangolo.com/tutorial/request-files/
@router.post("/upload-checkpoint-metadata/", summary="Upload metadata", response_description="Filename on success")
async def upload_checkpoint_metadata(new_file: Union[UploadFile, None] = None):
    """
    Upload the LSTM checkpoint metadata file.

    - **new_file**: Required.  The file to upload.
    """

    output_filename = "checkpoint"
    output_file = os.path.join(BASE_CKPT_DIR, output_filename)

    if not new_file:
        return NO_FILE_ERROR_RESPONSE
    else:
        if not os.path.isdir(BASE_CKPT_DIR):
            os.makedirs(BASE_CKPT_DIR)

        async with open_aio(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

@router.post("/upload-checkpoint-index/", summary="Upload index", response_description="Filename on success")
async def upload_checkpoint_index(new_file: Union[UploadFile, None] = None):
    """
    Upload the LSTM checkpoint index file.

    - **new_file**: Required.  The file to upload.
    """

    output_filename = "my_ckpt.index"
    output_file = os.path.join(BASE_CKPT_DIR, output_filename)

    if not new_file:
        return NO_FILE_ERROR_RESPONSE
    else:
        if not os.path.isdir(BASE_CKPT_DIR):
            os.makedirs(BASE_CKPT_DIR)

        async with open_aio(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

@router.post("/upload-checkpoint-data/", summary="Upload checkpoint data", response_description="Filename on success")
async def upload_checkpoint_data(new_file: Union[UploadFile, None] = None):
    """
    Upload the LSTM checkpoint data file.

    - **new_file**: Required.  The file to upload.
    """

    output_filename = "my_ckpt.data-00000-of-00001"
    output_file = os.path.join(BASE_CKPT_DIR, output_filename)

    if not new_file:
        return NO_FILE_ERROR_RESPONSE
    else:
        if not os.path.isdir(BASE_CKPT_DIR):
            os.makedirs(BASE_CKPT_DIR)

        async with open_aio(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

@router.post("/upload-train-data/", summary="Upload training data", response_description="Filename on success")
async def upload_train_data(new_file: Union[UploadFile, None] = None):
    """
    Upload the LSTM checkpoint training data csv file.

    - **new_file**: Required.  The file to upload.
    """

    output_filename = "train_data.csv"
    output_file = os.path.join(BASE_CKPT_DIR, output_filename)

    if not new_file:
        return NO_FILE_ERROR_RESPONSE
    else:
        if not os.path.isdir(BASE_CKPT_DIR):
            os.makedirs(BASE_CKPT_DIR)

        async with open_aio(output_file, 'wb') as out_file:
            while content := await new_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        return {"filename": new_file.filename}

@router.post("/refresh-model/", summary="Refresh model", response_description="Success")
def refresh_model(model: LstmStressClassifierModel = Depends(get_lstm_stress_classifier_model)):
    """
    Refresh the model to load new inputs
    """

    model.refresh_model()
    model.load_weights()

    return {"success": True}
