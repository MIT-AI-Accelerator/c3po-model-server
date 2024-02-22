import json
from fastapi.testclient import TestClient
from ppg.core.config import OriginationEnum
from app.main import versioned_app
from app.core.config import get_acronym_dictionary

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

# test upload acronym list
def test_upload_acronym_dictionary():
    acronym_dictionary = dict({'PPG': 'Prototype Proving Ground (PPG)'})
    response = client.post("/v1/upload_acronym_dictionary", params={'acronym_dictionary': json.dumps(acronym_dictionary)})
    assert response.status_code == 200
    assert response.json() == acronym_dictionary
    assert get_acronym_dictionary() == acronym_dictionary
