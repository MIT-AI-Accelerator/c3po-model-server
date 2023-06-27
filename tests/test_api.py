from fastapi.testclient import TestClient
from app.main import versioned_app
from app.core.config import OriginationEnum

client = TestClient(versioned_app)

# Verify app versioning set to /v1
def test_v1_exists():
    response = client.get(
        "/v1/docs"
    )
    assert response.status_code == 200

# set originated_from for standard app usage
def test_set_originated_from_app():
    response = client.get("/v1/originated_from_app")
    data = response.json()
    assert data == OriginationEnum.ORIGINATED_FROM_APP
    assert response.status_code == 200

# set originated_from for cleanup of database test entries
def test_set_originated_from_test():
    response = client.get("/v1/originated_from_test")
    data = response.json()
    assert data == OriginationEnum.ORIGINATED_FROM_TEST
    assert response.status_code == 200
