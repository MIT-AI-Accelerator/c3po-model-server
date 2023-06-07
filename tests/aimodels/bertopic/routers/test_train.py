import os
from unittest.mock import MagicMock
import uuid
from fastapi.testclient import TestClient
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel, EmbeddingModelTypeEnum
from app.aimodels.bertopic.models.document import DocumentModel

from app.main import app
from app.aimodels.bertopic.routers.train import get_db, get_minio
from tests.test_files.db.db_test_session import SessionLocal

from fastapi.encoders import jsonable_encoder

from sqlalchemy.orm import Session
from minio import Minio


# ************Mocks*******************
def mock_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

mock_s3 = MagicMock()
def mock_get_minio():
    return mock_s3

app.dependency_overrides = {get_db: mock_db, get_minio: mock_get_minio}
# *************************************


# test train endpoint with invalid request
def test_train_invalid_request(client: TestClient):

    body = {
        "wrong_param": '',
    }

    response = client.post(
        "/aimodels/bertopic/train",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422


# test train endpoint with valid request
def test_train_valid_request(client: TestClient, db: Session):
    pass

    # # create bertopic embedding pretrained object
    # embedding_pretrained_obj = BertopicEmbeddingPretrainedModel(id=uuid.uuid4(), sha256=valid_sha256)

    # # mock crud.bertopic_embedding_pretrained.get(db, request.bertopic_embedding_pretrained_id) to return the object created above
    # with mock.patch('app.aimodels.bertopic.crud.crud_bertopic_embedding_pretrained.bertopic_embedding_pretrained.get') as mock_get:
    #     mock_get.return_value = embedding_pretrained_obj

    # sentence_transformer_db = db.query(BertopicEmbeddingPretrainedModel).filter(
    #     BertopicEmbeddingPretrainedModel.model_type == EmbeddingModelTypeEnum.SENTENCE_TRANSFORMERS,
    #     BertopicEmbeddingPretrainedModel.uploaded == True).first()

    # documents_db = db.query(DocumentModel).limit(10).all()

    # body = {
    #     "sentence_transformer_id": sentence_transformer_db.id,
    #     "document_ids": [d.id for d in documents_db]
    # }

    # response = client.post(
    #     "/aimodels/bertopic/train",
    #     headers={},
    #     json=jsonable_encoder(body),
    # )

    # assert response.status_code == 200
    # assert response.json()['id'] is not None


# test train endpoint with valid request
def test_train_valid_request_weak_learning(client: TestClient, db: Session):
    pass

    # sentence_transformer_db = db.query(BertopicEmbeddingPretrainedModel).filter(
    #     BertopicEmbeddingPretrainedModel.model_type == EmbeddingModelTypeEnum.SENTENCE_TRANSFORMERS,
    #     BertopicEmbeddingPretrainedModel.uploaded == True).first()

    # weak_learner_db = db.query(BertopicEmbeddingPretrainedModel).filter(
    #     BertopicEmbeddingPretrainedModel.model_type == EmbeddingModelTypeEnum.WEAK_LEARNERS,
    #     BertopicEmbeddingPretrainedModel.uploaded == True).first()

    # documents_db = db.query(DocumentModel).limit(10).all()

    # body = {
    #     "sentence_transformer_id": sentence_transformer_db.id,
    #     "weak_learner_id": weak_learner_db.id,
    #     "document_ids": [d.id for d in documents_db]
    # }

    # response = client.post(
    #     "/aimodels/bertopic/train",
    #     headers={},
    #     json=jsonable_encoder(body),
    # )

    # assert response.status_code == 200
    # assert response.json()['id'] is not None
    # assert response.json() == {}
