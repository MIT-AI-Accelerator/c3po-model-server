import os
from unittest import mock
import uuid
from fastapi.testclient import TestClient
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.aimodels.bertopic.schemas.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate

from app.main import app
from app.aimodels.bertopic.routers.bertopic_embedding_pretrained import get_db
from tests.test_files.db.db_test_session import SessionLocal

from fastapi.encoders import jsonable_encoder

# ************Mocks*******************


def mock_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = mock_db
# *************************************


# test train endpoint with valid request
def test_train_valid_request(client: TestClient, valid_sha256: str):
    pass

    # # create bertopic embedding pretrained object
    # embedding_pretrained_obj = BertopicEmbeddingPretrainedModel(id=uuid.uuid4(), sha256=valid_sha256)

    # # mock crud.bertopic_embedding_pretrained.get(db, request.bertopic_embedding_pretrained_id) to return the object created above
    # with mock.patch('app.aimodels.bertopic.crud.crud_bertopic_embedding_pretrained.bertopic_embedding_pretrained.get') as mock_get:
    #     mock_get.return_value = embedding_pretrained_obj

    #     body = {
    #         "bertopic_embedding_pretrained_id": "1",
    #         "document_ids": ["1", "2"]
    #     }

    #     response = client.post(
    #         "/aimodels/bertopic/train",
    #         headers={},
    #         json=jsonable_encoder(body),
    #     )

    #     assert response.status_code == 200
    #     assert response.json()['id'] is not None
    #     assert response.json() == {}
