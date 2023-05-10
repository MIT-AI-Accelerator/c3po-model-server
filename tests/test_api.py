from fastapi.testclient import TestClient
from app.main import versioned_app

client = TestClient(versioned_app)

# Verify app versioning set to /v1
def test_v1_exists():
    response = client.get(
        "/v1/docs"
    )
    assert response.status_code == 200
