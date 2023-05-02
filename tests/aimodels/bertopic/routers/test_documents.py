
from fastapi.testclient import TestClient
from app.aimodels.bertopic.schemas.document import DocumentCreate

from app.main import app
from app.aimodels.bertopic.routers.documents import get_db
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
