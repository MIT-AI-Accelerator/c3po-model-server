import uuid, datetime
import pytest
import pandas as pd
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from _pytest.monkeypatch import MonkeyPatch
from pytest_mock import MockerFixture
from app.core.config import environment_settings, settings, OriginationEnum
from app.mattermost.crud import crud_mattermost
from app.mattermost.models.mattermost_channels import MattermostChannelModel
from app.mattermost.models.mattermost_users import MattermostUserModel
from app.mattermost.models.mattermost_documents import MattermostDocumentModel
from app.aimodels.bertopic.crud import crud_document
from app.ppg_common.schemas.bertopic.document import DocumentCreate
from app.ppg_common.schemas.mattermost.mattermost_documents import ThreadTypeEnum

@pytest.fixture(scope='module')
def channel_db_obj(db: Session):
    channel_info = dict(id=str(uuid.uuid4()),
                name=f'my channel doc {datetime.datetime.now()}',
                team_id=str(uuid.uuid4()),
                team_name=f'my team doc {datetime.datetime.now()}',
                display_name='my channel',
                type='P',
                header='my header',
                purpose='my purpose')
    return crud_mattermost.populate_mm_channel_info(db, channel_info=channel_info)

@pytest.fixture(scope='module')
def user_db_obj(channel_db_obj: MattermostChannelModel, db: Session):
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

@pytest.fixture(scope='module')
def mm_db_obj(channel_db_obj: MattermostChannelModel,
              user_db_obj: MattermostUserModel,
              db: Session):
    doc_obj_in = DocumentCreate(text='Spirit Bomb')
    doc_db_obj = crud_document.document.create(db, obj_in=doc_obj_in)
    mm_doc_obj_in = MattermostDocumentModel(message_id=str(uuid.uuid4()),
                                            root_message_id='',
                                            channel=channel_db_obj.id,
                                            user=user_db_obj.id,
                                            document=doc_db_obj.id,
                                            type='',
                                            hashtags='',
                                            props=dict(),
                                            doc_metadata=dict())
    return crud_mattermost.mattermost_documents.create(db, obj_in=mm_doc_obj_in)

@pytest.fixture(scope='module')
def mm_db_obj_thread(channel_db_obj: MattermostChannelModel,
                     user_db_obj: MattermostUserModel,
                     mm_db_obj: MattermostDocumentModel,
                     db: Session):
    doc_obj_in = DocumentCreate(text='Super Spirit Bomb')
    doc_db_obj = crud_document.document.create(db, obj_in=doc_obj_in)
    mm_doc_obj_in = MattermostDocumentModel(message_id=str(uuid.uuid4()),
                                            root_message_id='',
                                            channel=channel_db_obj.id,
                                            user=user_db_obj.id,
                                            document=doc_db_obj.id,
                                            type='',
                                            hashtags='',
                                            props=dict(),
                                            doc_metadata=dict(),
                                            thread_type=ThreadTypeEnum.THREAD)
    return crud_mattermost.mattermost_documents.create(db, obj_in=mm_doc_obj_in)

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
def test_upload_mattermost_user_info(client: TestClient, monkeypatch: MonkeyPatch):

    if environment_settings.environment == 'test':
        return

    # see note at tests/mattermost/test_mattermost_router.py::test_get_mattermost_user_info
    monkeypatch.setattr(settings, 'originated_from', OriginationEnum.ORIGINATED_FROM_APP)

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
def test_get_mattermost_user_info(client: TestClient, monkeypatch: MonkeyPatch):

    if environment_settings.environment == 'test':
        return

    # This test creates db entries for mm user and channel; these are
    # operational entries and should not labeled as originated from test.
    # Use monkeypatch to reset global settings attribute upon test completion
    monkeypatch.setattr(settings, 'originated_from', OriginationEnum.ORIGINATED_FROM_APP)

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

def test_upload_mattermost_documents_valid_input(db: Session,
                                                 client: TestClient,
                                                 mocker: MockerFixture):
    # create a channel and user for upload
    channel_info = dict(id=str(uuid.uuid4()),
                    name=f'my channel upload {datetime.datetime.now()}',
                    team_id=str(uuid.uuid4()),
                    team_name=f'my team upload {datetime.datetime.now()}',
                    display_name='my channel',
                    type='P',
                    header='my header',
                    purpose='my purpose')
    channel_db_obj = crud_mattermost.populate_mm_channel_info(db, channel_info=channel_info)
    user = str(uuid.uuid4())
    mm_user = dict(id=user,
                   username=user,
                   nickname=user,
                   first_name='Gohan',
                   last_name='Son',
                   position='Saiyaman',
                   email='%s@nitmre.mil' % user)
    teams = {'0': 'a team', '1': 'b team'}
    user_db_obj = crud_mattermost.populate_mm_user_info(db, mm_user=mm_user, teams=teams)

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
    mocker.patch('app.ppg_common.services.mattermost_utils.get_channel_posts', return_value=mock_data)

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

def test_get_mattermost_documents_no_document(db: Session, client: TestClient):
    # create a channel and user without a document
    channel_info = dict(id=str(uuid.uuid4()),
                    name=f'my channel no doc {datetime.datetime.now()}',
                    team_id=str(uuid.uuid4()),
                    team_name=f'my team no doc {datetime.datetime.now()}',
                    display_name='my channel',
                    type='P',
                    header='my header',
                    purpose='my purpose')
    channel_db_obj = crud_mattermost.populate_mm_channel_info(db, channel_info=channel_info)
    user = str(uuid.uuid4())
    mm_user = dict(id=user,
                   username=user,
                   nickname=user,
                   first_name='Gohan',
                   last_name='Son',
                   position='Saiyaman',
                   email='%s@nitmre.mil' % user)
    teams = {'0': 'a team', '1': 'b team'}
    user_db_obj = crud_mattermost.populate_mm_user_info(db, mm_user=mm_user, teams=teams)

    response = client.get('/mattermost/documents/get',
                          headers={},
                          params={'team_name': channel_db_obj.team_name,
                                  'channel_name': channel_db_obj.channel_name})

    assert response.status_code == 422
    assert 'documents not found' in response.json()['detail']

def test_get_mattermost_documents_valid(channel_db_obj: MattermostChannelModel,
                                        mm_db_obj: MattermostDocumentModel,
                                        db: Session,
                                        client: TestClient):
    response = client.get('/mattermost/documents/get',
                          headers={},
                          params={'team_name': channel_db_obj.team_name,
                                  'channel_name': channel_db_obj.channel_name})

    mm_docs = response.json()

    assert response.status_code == 200
    assert str(mm_db_obj.id) in [doc['id'] for doc in mm_docs]

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

def test_mattermost_conversation_thread_no_thread(mm_db_obj: MattermostDocumentModel,
                                                 client: TestClient):
    response = client.post('mattermost/conversation_threads',
                           headers={},
                           json={'mattermost_document_ids': [str(mm_db_obj.id)]})

    mm_docs = response.json()

    assert response.status_code == 200
    assert str(mm_db_obj.message_id) in [mm_doc['message_id'] for mm_doc in mm_docs['threads']]
    assert str(mm_db_obj.message_id) in [mm_doc['message_id'] for mm_doc in mm_docs['threads_speaker']]
    assert str(mm_db_obj.message_id) in [mm_doc['message_id'] for mm_doc in mm_docs['threads_speaker_persona']]

def test_mattermost_conversation_thread_thread(mm_db_obj_thread: MattermostDocumentModel,
                                               client: TestClient):
    response = client.post('mattermost/conversation_threads',
                           headers={},
                           json={'mattermost_document_ids': [str(mm_db_obj_thread.id)]})

    mm_docs = response.json()

    assert response.status_code == 200
    assert str(mm_db_obj_thread.message_id) in [mm_doc['message_id'] for mm_doc in mm_docs['threads']]
    assert str(mm_db_obj_thread.message_id) in [mm_doc['message_id'] for mm_doc in mm_docs['threads_speaker']]
    assert str(mm_db_obj_thread.message_id) in [mm_doc['message_id'] for mm_doc in mm_docs['threads_speaker_persona']]
