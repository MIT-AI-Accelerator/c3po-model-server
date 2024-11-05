from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder

from app.core.config import OriginationEnum
from app.ppg_common.schemas.bertopic.document import DocumentCreate


def test_create_document_object_post_valid_request(client: TestClient):
    body = [DocumentCreate(
        text="This is a test document",
    )]

    response = client.post(
        "/aimodels/bertopic/documents",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()[0]['id'] is not None
    assert response.json()[0]['original_created_time'] is not None
    assert response.json()[0]['text'] == "This is a test document"

    # extend test_set_originated_from_* to confirm originated_from set properly in db
    # originated_from_test called in pytest client fixture
    assert response.json()[0]['originated_from'] == OriginationEnum.ORIGINATED_FROM_TEST

def test_create_document_object_post_invalid_request(client: TestClient):
    body = None

    response = client.post(
        "/aimodels/bertopic/documents",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422

def test_create_document_object_post_invalid_request_empty_list(client: TestClient):
    body = []

    response = client.post(
        "/aimodels/bertopic/documents",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json() == []
