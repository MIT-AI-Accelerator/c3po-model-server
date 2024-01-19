import hashlib
import os
from unittest.mock import MagicMock
import uuid
from sqlalchemy.orm import Session
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.core.config import OriginationEnum
from app.main import versioned_app
from app.aimodels.gpt4all.schemas.gpt4all_pretrained import Gpt4AllPretrainedCreate
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllModelFilenameEnum
import app.aimodels.gpt4all.crud.crud_gpt4all_pretrained as crud
from fastapi.encoders import jsonable_encoder

main_client = TestClient(versioned_app)

def test_create_gpt4all_pretrained_object_post_valid_request(valid_sha256: str):
    body = Gpt4AllPretrainedCreate(sha256=valid_sha256)

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['id'] is not None
    assert response.json()['uploaded'] == False
    assert type(response.json()['version']) == int
    assert response.json()['sha256'] == valid_sha256


def test_create_gpt4all_pretrained_object_post_invalid_request():
    body = {}

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422


def test_create_gpt4all_pretrained_object_post_invalid_request_sha256():
    body = {'sha256': '', 'model_name': 'test'}

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422
    assert response.json() == {'detail': [{'loc': [
        'body', 'sha256'], 'msg': 'sha256 must be hexademical and 64 characters long', 'type': 'value_error'}]}


def test_create_gpt4all_pretrained_object_post_already_existing_sha256(valid_sha256: str):
    body = Gpt4AllPretrainedCreate(sha256=valid_sha256)

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 400
    assert response.json() == {'detail': 'sha256 already exists'}


def test_create_gpt4all_pretrained_object_post_sha256_converted_to_lowercase(valid_sha256: str):
    body = Gpt4AllPretrainedCreate(sha256=valid_sha256.upper())

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['sha256'] == valid_sha256.lower()

# ************ upload ************


def test_upload_gpt4all_pretrained_object_post_valid_request(mocker):

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

    # create the gpt4all Embedding Pretrained Model object
    body = Gpt4AllPretrainedCreate(sha256=sha256_hash.hexdigest())

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    # Upload the file to the gpt4all Embedding Pretrained Model object
    with open(test_file, "rb") as f:
        mock_upload_file_to_minio = MagicMock(return_value=True)
        mocker.patch("app.aimodels.gpt4all.routers.pretrained.upload_file_to_minio",
                     new=mock_upload_file_to_minio)

        response2 = main_client.post(
            f"/backend/aimodels/gpt4all/pretrained/{embedding_pretrained_id}/upload/", files={"new_file": f})

        mock_upload_file_to_minio.assert_called_once()

    os.remove(test_file)

    assert response2.status_code == 200
    assert response2.json()["uploaded"] is True

# test upload with sha256 not matching the one in the database
def test_upload_gpt4all_pretrained_object_post_invalid_sha256(valid_sha256: str, mocker):
    # Create a file to upload
    test_file = "test_file_invalid_sha256"
    with open(test_file, "wb") as f:
        # generate a random string with negligible probability of collision
        contents = str(uuid.uuid4())
        f.write(contents.encode('utf-8'))

    # create the gpt4all Embedding Pretrained Model object
    body = Gpt4AllPretrainedCreate(sha256=valid_sha256)

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    # Upload the file to the gpt4all Embedding Pretrained Model object
    with open(test_file, "rb") as f:
        mock_upload_file_to_minio = MagicMock(return_value=True)
        mocker.patch("app.aimodels.gpt4all.routers.pretrained.upload_file_to_minio",
                     new=mock_upload_file_to_minio)

        response2 = main_client.post(
            f"/backend/aimodels/gpt4all/pretrained/{embedding_pretrained_id}/upload/", files={"new_file": f})

        mock_upload_file_to_minio.assert_not_called()

    os.remove(test_file)

    assert response2.status_code == 422
    assert response2.json() == {'detail': 'SHA256 hash mismatch'}


def test_upload_gpt4all_pretrained_object_post_empty_file(valid_sha256: str):
    body = Gpt4AllPretrainedCreate(sha256=valid_sha256)

    response = main_client.post(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    response2 = main_client.post(
        f"/backend/aimodels/gpt4all/pretrained/{embedding_pretrained_id}/upload/", files={"new_file": None})

    assert response2.status_code == 400


def test_upload_gpt4all_pretrained_object_post_invalid_id(mocker):
    test_file = "test_file_invalid_id"
    with open(test_file, "wb") as f:
        f.write(b"test data")

    with open(test_file, "rb") as f:
        mock_upload_file_to_minio = MagicMock(return_value=True)
        mocker.patch("app.aimodels.gpt4all.routers.pretrained.upload_file_to_minio",
                     new=mock_upload_file_to_minio)

        response = main_client.post(
            f"/backend/aimodels/gpt4all/pretrained/999/upload/", files={"new_file": f})

        mock_upload_file_to_minio.assert_not_called()

    os.remove(test_file)

    assert response.status_code == 422
    assert response.json() == {'detail': [{'loc': [
        'path', 'id'], 'msg': 'value is not a valid uuid', 'type': 'type_error.uuid'}]}

def test_get_gpt4all_pretrained_object_invalid_name():
    body = {'model_type': 'not_a_name.bin'}

    response = main_client.get(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        params=jsonable_encoder(body)
    )

    assert response.status_code == 422
    assert 'value is not a valid enumeration' in response.json()['detail'][0]['msg']

def test_get_gpt4all_pretrained_object_valid_name(client: TestClient, db: Session, mocker: MagicMock):
    body = {'model_type': Gpt4AllModelFilenameEnum.L13B_SNOOZY}

    mocked_model = crud.gpt4all_pretrained.get_latest_uploaded_by_model_type(db,
                                                                        model_type=Gpt4AllModelFilenameEnum.L13B_SNOOZY,
                                                                        originated_from=OriginationEnum.ORIGINATED_FROM_TEST)
    mocker.patch(
        "app.aimodels.gpt4all.crud.crud_gpt4all_pretrained.gpt4all_pretrained.get_latest_uploaded_by_model_type",
        return_value=mocked_model
    )

    response = main_client.get(
        "/backend/aimodels/gpt4all/pretrained",
        headers={},
        params=jsonable_encoder(body)
    )

    assert response.status_code == 200
