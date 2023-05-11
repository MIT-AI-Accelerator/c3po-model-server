import os
from unittest import mock
from fastapi.testclient import TestClient
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


def test_create_bertopic_embedding_pretrained_object_post_valid_request(client: TestClient, valid_sha256: str):
    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256)

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['id'] is not None
    assert response.json()['uploaded'] == False
    assert type(response.json()['version']) == int
    assert response.json()['sha256'] == valid_sha256


def test_create_bertopic_embedding_pretrained_object_post_invalid_request(client: TestClient):
    body = {}

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422


def test_create_bertopic_embedding_pretrained_object_post_invalid_request_sha256(client: TestClient):
    body = {'sha256': ''}

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422
    assert response.json() == {'detail': [{'loc': [
        'body', 'sha256'], 'msg': 'sha256 must be hexademical and 64 characters long', 'type': 'value_error'}]}

def test_create_bertopic_embedding_pretrained_object_post_already_existing_sha256(client: TestClient, valid_sha256: str):
    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256)

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 400
    assert response.json() == {'detail': [
        {'loc': ['body', 'sha256'], 'msg': 'sha256 already exists', 'type': 'value_error'}]}


def test_create_bertopic_embedding_pretrained_object_post_sha256_converted_to_lowercase(client: TestClient, valid_sha256: str):
    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256.upper())

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['sha256'] == valid_sha256.lower()

# ************ upload ************


def test_upload_bertopic_embedding_pretrained_object_post_valid_request(client: TestClient, valid_sha256: str):
    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256)

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    # Upload a file to the BERTopic Embedding Pretrained Model object
    test_file = "test_file.pkl"
    with open(test_file, "wb") as f:
        f.write(b"test data")

    with mock.patch("app.aimodels.bertopic.routers.bertopic_embedding_pretrained.BASE_CKPT_DIR", "."):
        with open(test_file, "rb") as f:
            response2 = client.post(
                f"/aimodels/bertopic/bertopic-embedding-pretrained/{embedding_pretrained_id}/upload/", files={"new_file": f})

    # delete the file
    os.remove(test_file)

    assert response2.status_code == 200
    assert response2.json()["uploaded"] is True


def test_upload_bertopic_embedding_pretrained_object_post_empty_file(client: TestClient, valid_sha256: str):
    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256)

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    response2 = client.post(
        f"/aimodels/bertopic/bertopic-embedding-pretrained/{embedding_pretrained_id}/upload/", files={"new_file": None})

    assert response2.status_code == 400


def test_upload_bertopic_embedding_pretrained_object_post_invalid_id(client: TestClient):
    test_file = "test_file.pkl"
    with open(test_file, "wb") as f:
        f.write(b"test data")

    with mock.patch("app.aimodels.bertopic.routers.bertopic_embedding_pretrained.BASE_CKPT_DIR", "."):
        with open(test_file, "rb") as f:
            response = client.post(
                f"/aimodels/bertopic/bertopic-embedding-pretrained/999/upload/", files={"new_file": f})

    # delete the file
    os.remove(test_file)

    assert response.status_code == 422
