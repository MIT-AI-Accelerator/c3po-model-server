import hashlib
from typing import Union
from fastapi import Depends, APIRouter, UploadFile

from fastapi import HTTPException

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import UUID4

from sqlalchemy.orm import Session
from minio import Minio
from ppg.schemas.bertopic.bertopic_embedding_pretrained import BertopicEmbeddingPretrained, BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate
from app.core.minio import upload_file_to_minio
from app.core.errors import HTTPValidationError, ValidationError
from app.dependencies import get_db, get_minio
from .. import crud
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel


router = APIRouter(
    prefix="/bertopic-embedding-pretrained"
)


@router.post(
    "/",
    response_model=Union[BertopicEmbeddingPretrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Create BERTopic Embedding Pretrained Model object",
    response_description="Created Embedding Pretrained Model object"
)
def create_bertopic_embedding_pretrained_object_post(bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedCreate, db: Session = Depends(get_db)) -> (
    Union[BertopicEmbeddingPretrained, HTTPValidationError]
):
    """
    Create BERTopic Embedding Pretrained Model object.
    """

    # check to make sure sha256 doesn't already exist
    obj_by_sha: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get_by_sha256(
        db, sha256=bertopic_embedding_pretrained_obj.sha256)

    if obj_by_sha:
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder(HTTPValidationError(detail=[ValidationError(loc=['body', 'sha256'], msg='sha256 already exists', type='value_error')]))
        )

    # pydantic handles validation
    new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.create(
        db, obj_in=bertopic_embedding_pretrained_obj)

    return new_bertopic_embedding_pretrained_obj

@router.post(
    "/{id}/upload/",
    response_model=Union[BertopicEmbeddingPretrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Upload BERTopic Embedding Pretrained Model Binary",
    response_description="Uploaded Embedding Pretrained Model Binary"
)
async def upload_bertopic_embedding_post(new_file: UploadFile, id: UUID4, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    Union[BertopicEmbeddingPretrained, HTTPValidationError]
):
    """
    Upload BERTopic Embedding Pretrained Model Binary.

    - **new_file**: Required.  The file to upload.
    """

    # check to make sure id exists
    bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get(
        db, id)
    if not bertopic_embedding_pretrained_obj:
        raise HTTPException(status_code=422, detail="BERTopic Embedding Pretrained Model not found")

    # check for hash
    if not bertopic_embedding_pretrained_obj.sha256:
        raise HTTPException(status_code=422, detail="BERTopic Embedding Pretrained Model hash not found")

    # validate sha256 hash against file
    sha256_hash = hashlib.sha256()
    while chunk := await new_file.read(8192):
        sha256_hash.update(chunk)
    if sha256_hash.hexdigest() != bertopic_embedding_pretrained_obj.sha256:
        raise HTTPException(status_code=422, detail="SHA256 hash mismatch")

    # upload to minio
    upload_file_to_minio(file=new_file, id=id, s3=s3)

    # update the object in db and return
    new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.update(
        db, db_obj=bertopic_embedding_pretrained_obj, obj_in=BertopicEmbeddingPretrainedUpdate(uploaded=True))

    return new_bertopic_embedding_pretrained_obj

@router.get(
    "/",
    response_model=Union[BertopicEmbeddingPretrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Get latest uploaded BERTopic Embedding Pretrained Model object",
    response_description="Retrieved latest Embedding Pretrained Model object"
)
def get_latest_bertopic_embedding_pretrained_object(model_name: str, db: Session = Depends(get_db)) -> (
    Union[BertopicEmbeddingPretrained, HTTPValidationError]
):
    """
    Get latest uploaded BERTopic Embedding Pretrained Model object.
    """
    bertopic_embedding_pretrained_obj = crud.bertopic_embedding_pretrained.get_by_model_name(
        db, model_name=model_name)

    if not bertopic_embedding_pretrained_obj:
        raise HTTPException(status_code=422, detail="BERTopic Embedding Pretrained Model not found")

    return bertopic_embedding_pretrained_obj
