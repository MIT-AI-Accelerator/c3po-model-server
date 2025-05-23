import uuid
import hashlib
import os
import pytest
from sqlalchemy.orm import Session
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder
from app.core.config import OriginationEnum
from app.ppg_common.schemas.gpt4all.llm_pretrained import LlmPretrainedCreate
from app.aimodels.gpt4all.models.llm_pretrained import LlmFilenameEnum
import app.aimodels.gpt4all.crud.crud_llm_pretrained as crud


def test_create_llm_pretrained_object_post_valid_request(client: TestClient, valid_sha256: str):
    body = LlmPretrainedCreate(sha256=valid_sha256)

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['id'] is not None
    assert response.json()['uploaded'] == False
    assert type(response.json()['version']) == int
    assert response.json()['sha256'] == valid_sha256


def test_create_llm_pretrained_object_post_invalid_request(client: TestClient):
    body = {}

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422


def test_create_llm_pretrained_object_post_invalid_request_sha256(client: TestClient):
    body = {'sha256': '', 'model_name': 'test'}

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'Value error, sha256 must be hexademical and 64 characters long'


def test_create_llm_pretrained_object_post_already_existing_sha256(client: TestClient, valid_sha256: str):
    body = LlmPretrainedCreate(sha256=valid_sha256)

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 400
    assert response.json() == {'detail': 'sha256 already exists'}


def test_create_llm_pretrained_object_post_sha256_converted_to_lowercase(client: TestClient, valid_sha256: str):
    body = LlmPretrainedCreate(sha256=valid_sha256.upper())

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['sha256'] == valid_sha256.lower()

# ************ upload ************


@pytest.mark.parametrize('model_type', [e for e in LlmFilenameEnum])
def test_upload_llm_pretrained_object_post_valid_request(client: TestClient,
                                                         mocker: MockerFixture,
                                                         model_type: LlmFilenameEnum):

    # Create a file to upload
    test_file = "test_file"
    with open(test_file, "wb") as f:
        # generate a random string with negligible probability of collision
        contents = str(uuid.uuid4())
        f.write(contents.encode('utf-8'))

    sha256_hash = hashlib.sha256()
    with open(test_file, "rb") as f:
        # compute the sha256 hash of the file
        while chunk := f.read(8192):
            sha256_hash.update(chunk)

    # create the LLM Embedding Pretrained Model object
    body = LlmPretrainedCreate(sha256=sha256_hash.hexdigest(),
                               model_type=model_type)

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    # Upload the file to the LLM Embedding Pretrained Model object
    with open(test_file, "rb") as f:
        mock_upload_file_to_s3 = MagicMock(return_value=True)
        mocker.patch("app.aimodels.gpt4all.routers.pretrained.upload_file_to_s3",
                     new=mock_upload_file_to_s3)

        response2 = client.post(
            f"/aimodels/llm/pretrained/{embedding_pretrained_id}/upload/", files={"new_file": f})

        mock_upload_file_to_s3.assert_called_once()

    os.remove(test_file)

    assert response2.status_code == 200
    assert response2.json()["uploaded"] is True


# test upload with sha256 not matching the one in the database
def test_upload_llm_pretrained_object_post_invalid_sha256(client: TestClient, valid_sha256: str, mocker):
    # Create a file to upload
    test_file = "test_file_invalid_sha256"
    with open(test_file, "wb") as f:
        # generate a random string with negligible probability of collision
        contents = str(uuid.uuid4())
        f.write(contents.encode('utf-8'))

    # create the gpt4all Embedding Pretrained Model object
    body = LlmPretrainedCreate(sha256=valid_sha256)

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    # Upload the file to the gpt4all Embedding Pretrained Model object
    with open(test_file, "rb") as f:
        mock_upload_file_to_s3 = MagicMock(return_value=True)
        mocker.patch("app.aimodels.gpt4all.routers.pretrained.upload_file_to_s3",
                     new=mock_upload_file_to_s3)

        response2 = client.post(
            f"/aimodels/llm/pretrained/{embedding_pretrained_id}/upload/", files={"new_file": f})

        mock_upload_file_to_s3.assert_not_called()

    os.remove(test_file)

    assert response2.status_code == 422
    assert response2.json() == {'detail': 'SHA256 hash mismatch'}


def test_upload_llm_pretrained_object_post_empty_file(client: TestClient, valid_sha256: str):
    body = LlmPretrainedCreate(sha256=valid_sha256)

    response = client.post(
        "/aimodels/llm/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    response2 = client.post(
        f"/aimodels/llm/pretrained/{embedding_pretrained_id}/upload/", files={"new_file": None})

    assert response2.status_code == 400


def test_upload_llm_pretrained_object_post_invalid_id(client: TestClient, mocker):
    test_file = "test_file_invalid_id"
    with open(test_file, "wb") as f:
        f.write(b"test data")

    with open(test_file, "rb") as f:
        mock_upload_file_to_s3 = MagicMock(return_value=True)
        mocker.patch("app.aimodels.gpt4all.routers.pretrained.upload_file_to_s3",
                     new=mock_upload_file_to_s3)

        response = client.post(
            f"/aimodels/llm/pretrained/999/upload/", files={"new_file": f})

        mock_upload_file_to_s3.assert_not_called()

    os.remove(test_file)

    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'Input should be a valid UUID, invalid length: expected length 32 for simple format, found 3'


def test_upload_llm_pretrained_object_post_id_does_not_exist(client: TestClient):
    test_file = 'test_file_id_does_not_exist'
    with open(test_file, 'wb') as f:
        f.write(b'test data')

    with open(test_file, 'rb') as f:
        response = client.post('/aimodels/llm/pretrained/%s/upload/' % str(uuid.uuid4()),
                               files={'new_file': f})

    # the file persists if this test is run independently
    os.remove(test_file)

    assert response.status_code == 422
    assert 'gpt4all Pretrained Model not found' in response.json()['detail']


def test_get_llm_pretrained_object_invalid_name(client: TestClient):
    body = {'model_type': 'not_a_name.bin'}

    response = client.get(
        "/aimodels/llm/pretrained",
        headers={},
        params=jsonable_encoder(body)
    )

    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == "Input should be 'ggml-gpt4all-l13b-snoozy.bin' or 'mistrallite.Q4_K_M.gguf'"


@pytest.mark.parametrize('model_type', [e for e in LlmFilenameEnum])
def test_get_llm_pretrained_object_valid_name(client: TestClient,
                                              db: Session,
                                              mocker: MagicMock,
                                              model_type: LlmFilenameEnum):
    body = {'model_type': model_type}

    mocked_model = crud.llm_pretrained.get_latest_uploaded_by_model_type(
        db,
        model_type=model_type,
        originated_from=OriginationEnum.ORIGINATED_FROM_TEST)
    mocker.patch(
        "app.aimodels.gpt4all.crud.crud_llm_pretrained.llm_pretrained.get_latest_uploaded_by_model_type",
        return_value=mocked_model
    )

    response = client.get(
        "/aimodels/llm/pretrained",
        headers={},
        params=jsonable_encoder(body)
    )

    assert response.status_code == 200
