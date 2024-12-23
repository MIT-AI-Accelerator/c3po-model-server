import hashlib
import pickle
import pandas as pd
from typing import Union
from io import TextIOWrapper, StringIO, BytesIO

from fastapi import Depends, APIRouter, UploadFile
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from pydantic import UUID4, BaseModel, ConfigDict
from sqlalchemy.orm import Session
from minio import Minio

from app.core.minio import upload_file_to_minio
from app.core.errors import HTTPValidationError, ValidationError
from app.core.config import get_label_dictionary, set_label_dictionary
from app.dependencies import get_db, get_minio
from app.ppg_common.schemas.bertopic.bertopic_embedding_pretrained import BertopicEmbeddingPretrained, BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate, EmbeddingModelTypeEnum
from .. import crud
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from ..ai_services.weak_learning import WeakLearner


ERR_MSG_MODEL_NOT_FOUND = "BERTopic Embedding Pretrained Model not found"


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

    if bertopic_embedding_pretrained_obj.model_type == EmbeddingModelTypeEnum.WEAK_LEARNERS:
        bertopic_embedding_pretrained_obj.reference = get_label_dictionary()

    # pydantic handles validation
    new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.create(
        db, obj_in=bertopic_embedding_pretrained_obj)

    return new_bertopic_embedding_pretrained_obj

@router.post(
    "/train/",
    response_model=Union[list, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Train a Weak Learner Model for upload",
    response_description="Trained the Weak Learner Model"
)
async def upload_bertopic_embedding_post(new_file: UploadFile, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    Union[list, HTTPValidationError]
):
    """
    Train a Weak Learner Model for upload.

    - **new_file**: Required.  The training data.
    """

    # load the training data
    cstr = ""
    with new_file.file as f:
        for line in TextIOWrapper(f, encoding='utf-8'):
            cstr += line
    df_train = pd.read_csv(StringIO(cstr))

    # check for columns required to train weak learner
    if not set(['message', 'createat', 'labels']).issubset(df_train.columns):
        raise HTTPException(status_code=422, detail="unable to train weak learner")

    # train and serialize the weak learner object, return as binary
    weak_learners_model_obj = WeakLearner().train_weak_learners(df_train)
    bin_data = pickle.dumps(weak_learners_model_obj)
    response = StreamingResponse(BytesIO(bin_data), media_type="application/octet-stream")

    response.headers["Content-Disposition"] = f"attachment; filename={new_file.filename}.bin"
    return response

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
        raise HTTPException(status_code=422, detail=ERR_MSG_MODEL_NOT_FOUND)

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
        raise HTTPException(status_code=422, detail=ERR_MSG_MODEL_NOT_FOUND)

    return bertopic_embedding_pretrained_obj

@router.get(
    "/label-dictionary/get",
    response_model=Union[list, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Get label dictionary for latest uploaded BERTopic Weak Learner Model object",
    response_description="Retrieved label dictionary for latest uploaded BERTopic Weak Learner Model object"
)
def get_latest_weak_label_dictionary(model_name: str, db: Session = Depends(get_db)) -> (
    Union[list, HTTPValidationError]
):
    """
    Get label dictionary for latest uploaded BERTopic Weak Learner Model object.
    """
    bertopic_embedding_pretrained_obj = crud.bertopic_embedding_pretrained.get_by_model_name(
        db, model_name=model_name)

    if not bertopic_embedding_pretrained_obj:
        raise HTTPException(status_code=422, detail=ERR_MSG_MODEL_NOT_FOUND)

    if bertopic_embedding_pretrained_obj.model_type != EmbeddingModelTypeEnum.WEAK_LEARNERS:
        raise HTTPException(status_code=422, detail=f"Label dictionary not used for model type {bertopic_embedding_pretrained_obj.model_type}")

    if 'labeling_terms' not in bertopic_embedding_pretrained_obj.reference.keys():
        raise HTTPException(status_code=422, detail="Label dictionary not found")

    return bertopic_embedding_pretrained_obj.reference['labeling_terms']

class LabelDictionaryRequest(BaseModel):
    model_name: str = ""
    labeling_terms: list = []

    model_config = ConfigDict(protected_namespaces=())

@router.post(
    "/label-dictionary/append",
    response_model=Union[list, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Append to latest label dictionary, train and upload a new BERTopic Weak Learner Model object",
    response_description="Uploaded Embedding Pretrained Model Binary"
)
def append_latest_weak_label_dictionary(request: LabelDictionaryRequest, db: Session = Depends(get_db)) -> (
    Union[list, HTTPValidationError]
):
    """
    Append to latest label dictionary, train and upload a new BERTopic Weak Learner Model object.
    """
    bertopic_embedding_pretrained_obj = crud.bertopic_embedding_pretrained.get_by_model_name(
        db, model_name=request.model_name)

    if not bertopic_embedding_pretrained_obj:
        raise HTTPException(status_code=422, detail=ERR_MSG_MODEL_NOT_FOUND)

    if bertopic_embedding_pretrained_obj.model_type != EmbeddingModelTypeEnum.WEAK_LEARNERS:
        raise HTTPException(status_code=422, detail=f"Label dictionary not used for model type {bertopic_embedding_pretrained_obj.model_type}")

    if 'labeling_terms' not in bertopic_embedding_pretrained_obj.reference.keys():
        raise HTTPException(status_code=422, detail="Label dictionary not found")

    label_dictionary = bertopic_embedding_pretrained_obj.reference
    label_dictionary['labeling_terms'] = label_dictionary['labeling_terms'] + [request.labeling_terms]
    label_dictionary = set_label_dictionary(label_dictionary)

    return label_dictionary['labeling_terms']
