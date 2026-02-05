import hashlib
import os
import io
import uuid
import pickle
import string
import random
import pandas as pd
from sqlalchemy.orm import Session
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.core.config import OriginationEnum
from app.ppg_common.schemas.bertopic.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate
from app.aimodels.bertopic.crud.crud_bertopic_embedding_pretrained import bertopic_embedding_pretrained
from app.aimodels.bertopic.ai_services.weak_learning import WeakLearner
from fastapi.encoders import jsonable_encoder

def test_create_bertopic_embedding_pretrained_object_post_valid_request(client: TestClient, valid_sha256: str):

    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256, model_name='test')

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

    body = {'sha256': '', 'model_name': 'test'}

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'Value error, sha256 must be hexademical and 64 characters long'


def test_create_bertopic_embedding_pretrained_object_post_already_existing_sha256(client: TestClient, valid_sha256: str):

    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256, model_name='test')

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

    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256.upper(), model_name='test')

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['sha256'] == valid_sha256.lower()


# ************ upload ************


def test_upload_bertopic_embedding_pretrained_object_post_valid_request(client: TestClient, mocker: MagicMock):

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

    # create the BERTopic Embedding Pretrained Model object
    body = BertopicEmbeddingPretrainedCreate(sha256=sha256_hash.hexdigest(), model_name='test')

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    # Upload the file to the BERTopic Embedding Pretrained Model object
    with open(test_file, "rb") as f:
        mock_upload_file_to_s3 = MagicMock(return_value=True)
        mocker.patch("app.aimodels.bertopic.routers.bertopic_embedding_pretrained.upload_file_to_s3",
                     new=mock_upload_file_to_s3)

        response2 = client.post(
            f"/aimodels/bertopic/bertopic-embedding-pretrained/{embedding_pretrained_id}/upload/", files={"new_file": f})

        mock_upload_file_to_s3.assert_called_once()

    os.remove(test_file)

    assert response2.status_code == 200
    assert response2.json()["uploaded"] is True


# test upload with sha256 not matching the one in the database
def test_upload_bertopic_embedding_pretrained_object_post_invalid_sha256(client: TestClient, valid_sha256: str, mocker: MagicMock):

    # Create a file to upload
    test_file = "test_file_invalid_sha256"
    with open(test_file, "wb") as f:
        # generate a random string with negligible probability of collision
        contents = str(uuid.uuid4())
        f.write(contents.encode('utf-8'))

    # create the BERTopic Embedding Pretrained Model object
    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256, model_name='test')

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    # Upload the file to the BERTopic Embedding Pretrained Model object
    with open(test_file, "rb") as f:
        mock_upload_file_to_s3 = MagicMock(return_value=True)
        mocker.patch("app.aimodels.bertopic.routers.bertopic_embedding_pretrained.upload_file_to_s3",
                     new=mock_upload_file_to_s3)

        response2 = client.post(
            f"/aimodels/bertopic/bertopic-embedding-pretrained/{embedding_pretrained_id}/upload/", files={"new_file": f})

        mock_upload_file_to_s3.assert_not_called()

    os.remove(test_file)

    assert response2.status_code == 422
    assert response2.json() == {'detail': 'SHA256 hash mismatch'}



def test_upload_bertopic_embedding_pretrained_object_post_empty_file(client: TestClient, valid_sha256: str):

    body = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256, model_name='test')

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    response2 = client.post(
        f"/aimodels/bertopic/bertopic-embedding-pretrained/{embedding_pretrained_id}/upload/", files={"new_file": None})

    assert response2.status_code == 400



def test_upload_bertopic_embedding_pretrained_object_post_invalid_request(client: TestClient, mocker: MagicMock):

    test_file = "test_file_invalid_request"
    with open(test_file, "wb") as f:
        f.write(b"test data")

    with open(test_file, "rb") as f:
        mock_upload_file_to_s3 = MagicMock(return_value=True)
        mocker.patch("app.aimodels.bertopic.routers.bertopic_embedding_pretrained.upload_file_to_s3",
                     new=mock_upload_file_to_s3)

        response = client.post(
            f"/aimodels/bertopic/bertopic-embedding-pretrained/999/upload/", files={"new_file": f})

        mock_upload_file_to_s3.assert_not_called()

    os.remove(test_file)

    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'Input should be a valid UUID, invalid length: expected length 32 for simple format, found 3'


def test_upload_bertopic_embedding_pretrained_object_post_invalid_id(client: TestClient, mocker: MagicMock):

    test_file = 'test_file_invalid_id'
    with open(test_file, 'wb') as f:
        f.write(b'test data')

    with open(test_file, 'rb') as f:
        response = client.post('/aimodels/bertopic/bertopic-embedding-pretrained/%s/upload/' % str(uuid.uuid4()),
                               files={'new_file': f})

    # the file persists if this test is run independently
    os.remove(test_file)

    assert response.status_code == 422
    assert 'BERTopic Embedding Pretrained Model not found' in response.json()['detail']


def test_upload_bertopic_embedding_pretrained_weak_learner_object_post_valid_request(client: TestClient):

    # Create a file to upload
    df_train = pd.DataFrame({'createat': [0,0,0,0,0,0,0,0,0,0],
                             'message': [''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10)),
                                         ''.join(random.choices(string.ascii_lowercase, k=10))],
                                         'labels': [0,1,2,0,1,2,0,1,2,0]})

    weak_learner_model_obj = WeakLearner().train_weak_learners(df_train)

    # Serialize the object
    serialized_obj = pickle.dumps(weak_learner_model_obj)

    # Calculate the SHA256 hash of the serialized object
    hash_object = hashlib.sha256(serialized_obj)
    hex_dig = hash_object.hexdigest()

    # create the BERTopic Embedding Pretrained Model object
    body = BertopicEmbeddingPretrainedCreate(sha256=hex_dig, model_name='test', model_type='weak_learners')

    response = client.post(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        json=jsonable_encoder(body),
    )
    embedding_pretrained_id = response.json()["id"]

    # Dump the object to the file object
    file_obj = io.BytesIO()
    pickle.dump(weak_learner_model_obj, file_obj)
    file_obj.seek(0)

    # Upload the file to the BERTopic Embedding Pretrained Model object
    response2 = client.post(
        f"/aimodels/bertopic/bertopic-embedding-pretrained/{embedding_pretrained_id}/upload/", files={"new_file": file_obj})

    assert response2.status_code == 200
    assert response2.json()["uploaded"] is True



def test_get_latest_bertopic_embedding_pretrained_object_invalid_request(client: TestClient):
    body = {'wrong_param': 'does not matter'}

    response = client.get(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        params=jsonable_encoder(body)
    )

    assert response.status_code == 422


def test_get_latest_bertopic_embedding_pretrained_object_invalid_name(client: TestClient):
    body = {'model_name': 'nonexistent_model'}

    response = client.get(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        params=jsonable_encoder(body)
    )

    assert response.status_code == 422


def test_get_latest_bertopic_embedding_pretrained_object_valid_name(client: TestClient, db: Session, mocker: MagicMock):
    my_model = 'test'
    body = {'model_name': my_model}

    mocked_model = bertopic_embedding_pretrained.get_by_model_name(db,
                                                                   model_name=my_model,
                                                                   originated_from=OriginationEnum.ORIGINATED_FROM_TEST)
    mocker.patch(
        "app.aimodels.bertopic.crud.crud_bertopic_embedding_pretrained.bertopic_embedding_pretrained.get_by_model_name",
        return_value=mocked_model
    )

    response = client.get(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        params=jsonable_encoder(body)
    )

    assert response.status_code == 200
    assert response.json()['model_name'] == my_model
