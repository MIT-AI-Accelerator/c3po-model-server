import uuid, datetime
import pytest
import pandas as pd
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from ppg.core.config import OriginationEnum
from app.core.config import environment_settings
from app.mattermost.crud import crud_mattermost
from app.mattermost.models.mattermost_channels import MattermostChannelModel
from app.mattermost.models.mattermost_users import MattermostUserModel

@pytest.fixture(scope='module')
def channel_db_obj(db: Session) -> MattermostChannelModel:
    channel_info = dict(id=str(uuid.uuid4()),
                        name='my channel',
                        team_id=str(uuid.uuid4()),
                        team_name='my team',
                        display_name='my channel',
                        type='P',
                        header='my header',
                        purpose='my purpose')
    return crud_mattermost.populate_mm_channel_info(db, channel_info=channel_info)

@pytest.fixture(scope='module')
def user_db_obj(db: Session) -> MattermostUserModel:
    user = str(uuid.uuid4())
    mm_user = dict(id=user,
                   username=user,
                   nickname=user,
                   first_name='Gohan',
                   last_name='Son',
                   position='Saiyaman',
                   email='%s@nitmre.mil' % user)
    teams = {'0': 'a team', '1': 'b team'}
    return crud_mattermost.populate_mm_user_info(db, mm_user=mm_user, teams=teams)

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

def test_upload_mattermost_documents_valid_input(channel_db_obj: MattermostChannelModel,
                                                 user_db_obj: MattermostUserModel,
                                                 client: TestClient,
                                                 mocker: MockerFixture):
    data = {
        'id': [str(uuid.uuid4())],
        'create_at': [datetime.datetime.now()],
        'update_at': [datetime.datetime.now()],
        'edit_at': [0],
        'delete_at': [0],
        'is_pinned': [False],
        'user_id': [user_db_obj.user_id],
        'channel_id': [channel_db_obj.channel_id],
        'channel': [channel_db_obj.id],
        'root_id': [''],
        'original_id': [''],
        'message': ['Kamehameha'],
        'type': [''],
        'props': [dict()],
        'hashtags': [''],
        'pending_post_id': [''],
        'reply_count': [0],
        'last_reply_at': [0],
        'participants': [''],
        'metadata': [dict()],
        'has_reactions': [False],
        'file_ids': [list()],
        'datetime': [datetime.datetime.now()]
    }
    mock_data = pd.DataFrame(data)
    mocker.patch('ppg.services.mattermost_utils.get_channel_posts', return_value=mock_data)

    response = client.post(
        '/mattermost/documents/upload',
        headers={},
        json={'channel_ids': [channel_db_obj.channel_id]}
    )

    assert response.status_code == 200
    assert len(response.json()) > 0

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

def test_get_mattermost_documents_no_document(channel_db_obj, client: TestClient):
    response = client.get('/mattermost/documents/get',
                          headers={},
                          params={'team_name': channel_db_obj.team_name,
                                  'channel_name': channel_db_obj.channel_name})

    assert response.status_code == 422
    assert 'documents not found' in response.json()['detail']

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
