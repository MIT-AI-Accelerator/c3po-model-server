import os
from typing import Union
from fastapi import Depends, APIRouter, UploadFile
from pydantic import UUID4
from ..schemas.bertopic_embedding_pretrained import BertopicEmbeddingPretrained, BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate
from app.dependencies import get_db, get_minio
from sqlalchemy.orm import Session
from .. import crud
from ..ai_services.basic_inference import BASE_CKPT_DIR
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.core.errors import HTTPValidationError, ValidationError
from aiofiles import open as open_aio
from minio import Minio


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
        return HTTPValidationError(detail=[ValidationError(loc=['path', 'bertopic model upload'], msg='Invalid pretrained model id', type='value_error')])

    # upload to minio
    output_filename = "bertopic_embedding.pkl"
    output_file = os.path.join(BASE_CKPT_DIR, output_filename)

    if not os.path.isdir(BASE_CKPT_DIR):
        os.makedirs(BASE_CKPT_DIR)

    async with open_aio(output_file, 'wb') as out_file:
        while content := await new_file.read(1024):  # async read chunk
            await out_file.write(content)  # async write chunk

    # update the object in db and return
    new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.update(
        db, db_obj=bertopic_embedding_pretrained_obj, obj_in=BertopicEmbeddingPretrainedUpdate(uploaded=True))

    return new_bertopic_embedding_pretrained_obj
