import uuid
from fastapi.testclient import TestClient
from ppg.core.config import OriginationEnum
from app.core.config import environment_settings

# returns 422
def test_upload_mattermost_user_info_invalid_format(client: TestClient):
    response = client.post(
        "/mattermost/user/upload",
        headers={},
        json={"notafield": "notauser"}
    )

    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'

# returns 422
def test_upload_mattermost_user_info_invalid_input(client: TestClient):
    response = client.post(
        "/mattermost/user/upload",
        headers={},
        json={"user_name": "notauser"}
    )

    assert response.status_code == 422
    assert 'Mattermost' in response.json()['detail']

# test user upload endpoint
def test_upload_mattermost_user_info(client: TestClient):

    if environment_settings.environment == 'test':
        return

    response = client.post(
        "/mattermost/user/upload",
        headers={},
        json={"user_name": "nitmre-bot"}
    )

    assert response.status_code == 200

# returns 422
def test_get_mattermost_user_info_invalid_format(client: TestClient):
    response = client.get(
        "/mattermost/user/get",
        headers={},
        params={"notafield": "notauser"}
    )

    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'

# returns 422
def test_get_mattermost_user_info_invalid_input(client: TestClient):
    response = client.get(
        "/mattermost/user/get",
        headers={},
        params={"user_name": "notauser"}
    )

    assert response.status_code == 422
    assert 'Mattermost' in response.json()['detail']

# test user get endpoint
def test_get_mattermost_user_info(client: TestClient):

    if environment_settings.environment == 'test':
        return

    # this test creates db entries for mm user and channel, these are
    # operational entries and should not labeled as originated from test
    response = client.get("/originated_from_app/")
    originated_from = response.json()
    assert originated_from == OriginationEnum.ORIGINATED_FROM_APP

    response = client.get(
        "/mattermost/user/get",
        headers={},
        params={"user_name": "nitmre-bot"}
    )

    assert response.status_code == 200

# returns 422
def test_upload_mattermost_documents_invalid_format(client: TestClient):
    response = client.post(
        "/mattermost/documents/upload",
        headers={},
        json={"notafield": "notachannelid"}
    )

    assert response.status_code == 422
    assert 'Mattermost' in response.json()['detail']

# returns 422
def test_upload_mattermost_documents_invalid_input(client: TestClient):
    response = client.post(
        "/mattermost/documents/upload",
        headers={},
        json={"channel_ids": ["notachannelid"]}
    )

    assert response.status_code == 422
    assert 'Mattermost' in response.json()['detail']

# returns 422
def test_get_mattermost_documents_invalid_format(client: TestClient):
    response = client.get(
        "/mattermost/documents/get",
        headers={},
        params={"notafield": "notachannelid"}
    )

    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'

# returns 422
def test_get_mattermost_documents_invalid_input(client: TestClient):
    response = client.get(
        "/mattermost/documents/get",
        headers={},
        params={"team_name": "notachannelname", "channel_name": "notachannelname"}
    )

    assert response.status_code == 422
    assert 'Mattermost' in response.json()['detail']

# returns 422
def test_mattermost_conversation_thread_invalid_format(client: TestClient):
    response = client.post(
        "/mattermost/conversation_threads",
        headers={},
        json={"mattermost_document_ids": "notadocumentlist"}
    )

    assert response.status_code == 422
    assert 'value is not a valid list' in response.json()['detail'][0]['msg']

# returns 422
def test_mattermost_conversation_thread_invalid_input(client: TestClient):
    response = client.post(
        "/mattermost/conversation_threads",
        headers={},
        json={"mattermost_document_ids": [f"{uuid.uuid4()}"]}
    )

    assert response.status_code == 422
    assert 'Mattermost' in response.json()['detail']
