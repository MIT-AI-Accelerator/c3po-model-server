from fastapi.testclient import TestClient
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
