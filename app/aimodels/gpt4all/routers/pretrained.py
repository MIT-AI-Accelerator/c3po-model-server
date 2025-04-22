
import hashlib
from typing import Union
from aiofiles import open as open_aio
from fastapi import Depends, APIRouter, UploadFile, HTTPException
from minio import Minio
from pydantic import UUID4
from sqlalchemy.orm import Session

from app.core.s3 import upload_file_to_s3
from app.dependencies import get_db, get_minio
from app.core.errors import HTTPValidationError
from app.ppg_common.schemas.gpt4all.llm_pretrained import LlmPretrained, LlmPretrainedCreate, LlmPretrainedUpdate
from .. import crud
from ..models.llm_pretrained import LlmPretrainedModel, LlmFilenameEnum



router = APIRouter(
    prefix="/pretrained"
)


@router.post(
    "/",
    response_model=Union[LlmPretrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Create gpt4all Pretrained Model object",
    response_description="Created Pretrained Model object"
)
def create_llm_pretrained_object_post(llm_pretrained_obj: LlmPretrainedCreate, db: Session = Depends(get_db)) -> (
    Union[LlmPretrained, HTTPValidationError]
):
    """
    Create GPT4All Pretrained Model object.
    """

    # check to make sure sha256 doesn't already exist
    obj_by_sha: LlmPretrainedModel = crud.llm_pretrained.get_by_sha256(
        db, sha256=llm_pretrained_obj.sha256)

    if obj_by_sha:
        raise HTTPException(status_code=400, detail="sha256 already exists")

    # pydantic handles validation
    new_llm_pretrained_obj: LlmPretrainedModel = crud.llm_pretrained.create(
        db, obj_in=llm_pretrained_obj)

    return new_llm_pretrained_obj

@router.post(
    "/{id}/upload/",
    response_model=Union[LlmPretrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Upload gpt4all Pretrained Model Binary",
    response_description="Uploaded Pretrained Model Binary"
)
async def upload_gpt4all_post(new_file: UploadFile, id: UUID4, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    Union[LlmPretrained, HTTPValidationError]
):
    """
    Upload gpt4all Pretrained Model Binary.

    - **new_file**: Required.  The file to upload.
    """

    # check to make sure id exists
    llm_pretrained_obj: LlmPretrainedModel = crud.llm_pretrained.get(
        db, id)
    if not llm_pretrained_obj:
        raise HTTPException(status_code=422, detail="gpt4all Pretrained Model not found")

    # check for hash
    if not llm_pretrained_obj.sha256:
        raise HTTPException(status_code=422, detail="gpt4all Pretrained Model hash not found")

    # validate sha256 hash against file
    sha256_hash = hashlib.sha256()
    while chunk := await new_file.read(8192):
        sha256_hash.update(chunk)
    if sha256_hash.hexdigest() != llm_pretrained_obj.sha256:
        raise HTTPException(status_code=422, detail="SHA256 hash mismatch")

    # upload to minio
    upload_file_to_s3(file=new_file, id=id, s3=s3)

    # update the object in db and return
    new_llm_pretrained_obj: LlmPretrainedModel = crud.llm_pretrained.update(
        db, db_obj=llm_pretrained_obj, obj_in=LlmPretrainedUpdate(uploaded=True))

    return new_llm_pretrained_obj


@router.get(
    "/",
    response_model=Union[LlmPretrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Get latest uploaded LLM Pretrained Model object",
    response_description="Retrieved latest LLM Pretrained Model object"
)
def get_latest_llm_pretrained_object(model_type: LlmFilenameEnum =
                                         LlmFilenameEnum.L13B_SNOOZY,
                                         db: Session = Depends(get_db)) -> (
    Union[LlmPretrained, HTTPValidationError]
):
    """
    Get latest uploaded LLM Pretrained Model object.
    """
    llm_pretrained_obj = crud.llm_pretrained.get_latest_uploaded_by_model_type(
        db, model_type=model_type
    )

    if not llm_pretrained_obj:
        raise HTTPException(status_code=422, detail="LLM Pretrained Model not found")

    return llm_pretrained_obj
