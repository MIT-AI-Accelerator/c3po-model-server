import uuid, datetime
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
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
    assert response.json()['detail'][0]['msg'] == 'Field required'

# returns 422
def test_upload_mattermost_user_info_invalid_input(client: TestClient):

    if (environment_settings.environment == 'test') or (environment_settings.environment == 'integration'):
        return

    response = client.post(
        "/mattermost/user/upload",
        headers={},
        json={"user_name": "notauser"}
    )

    assert response.status_code == 422
    assert 'Mattermost' in response.json()['detail']

# test user upload endpoint
def test_upload_mattermost_user_info(client: TestClient, monkeypatch: MonkeyPatch):

    if (environment_settings.environment == 'test') or (environment_settings.environment == 'integration'):
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
    assert response.json()['detail'][0]['msg'] == 'Field required'

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

    if (environment_settings.environment == 'test') or (environment_settings.environment == 'integration'):
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

    if (environment_settings.environment == 'test') or (environment_settings.environment == 'integration'):
        return

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
    assert response.json()['detail'][0]['msg'] == 'Field required'

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
    assert 'Input should be a valid list' in response.json()['detail'][0]['msg']

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
                                                  db: Session,
                                                  client: TestClient):
    response = client.post('mattermost/conversation_threads',
                           headers={},
                           json={'mattermost_document_ids': [str(mm_db_obj.id)]})

    mm_docs = response.json()
    assert response.status_code == 200

    thread_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids = mm_docs['threads'])
    assert str(mm_db_obj.message_id) in [row['message_id'] for key, row in thread_df.iterrows()]

    thread_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids = mm_docs['threads_speaker'])
    assert str(mm_db_obj.message_id) in [row['message_id'] for key, row in thread_df.iterrows()]

    thread_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids = mm_docs['threads_speaker_persona'])
    assert str(mm_db_obj.message_id) in [row['message_id'] for key, row in thread_df.iterrows()]

def test_mattermost_conversation_thread_thread(mm_db_obj_thread: MattermostDocumentModel,
                                                  db: Session,
                                                  client: TestClient):
    response = client.post('mattermost/conversation_threads',
                           headers={},
                           json={'mattermost_document_ids': [str(mm_db_obj_thread.id)]})

    mm_docs = response.json()
    assert response.status_code == 200

    thread_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids = mm_docs['threads'])
    assert str(mm_db_obj_thread.message_id) in [row['message_id'] for key, row in thread_df.iterrows()]

    thread_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids = mm_docs['threads_speaker'])
    assert str(mm_db_obj_thread.message_id) in [row['message_id'] for key, row in thread_df.iterrows()]

    thread_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids = mm_docs['threads_speaker_persona'])
    assert str(mm_db_obj_thread.message_id) in [row['message_id'] for key, row in thread_df.iterrows()]

def test_upload_mattermost_docs_by_substring(db: Session, client: TestClient, mocker: MockerFixture):
    # Mock Mattermost API response for substring search
    mock_data = pd.DataFrame([
        {
            'id': str(uuid.uuid4()),
            'create_at': datetime.datetime.now(),
            'update_at': datetime.datetime.now(),
            'edit_at': 0,
            'delete_at': 0,
            'is_pinned': False,
            'user_id': str(uuid.uuid4()),
            'channel_id': str(uuid.uuid4()),
            'root_id': '',
            'original_id': '',
            'message': 'This is a test message containing the substring.',
            'type': '',
            'props': {},
            'hashtags': '',
            'pending_post_id': '',
            'reply_count': 0,
            'last_reply_at': 0,
            'participants': '',
            'metadata': {},
            'has_reactions': False,
            'file_ids': [],
            'datetime': datetime.datetime.now(),
        }
    ])
    mocker.patch(
        'app.ppg_common.services.mattermost_utils.get_all_team_posts_by_substring',
        return_value=mock_data
    )

    # Create mock data for MattermostChannelModel
    channel = str(uuid.uuid4())
    mock_channel = MattermostChannelModel(
        id=channel,
        channel_id=str(uuid.uuid4()),
        channel_name=channel,
        team_id=str(uuid.uuid4()),
        team_name=channel,
        display_name='display_name',
        type='P',
        header='header',
        purpose='purpose',
        originated_from=OriginationEnum.ORIGINATED_FROM_TEST
    )
    db.add(mock_channel)
    db.commit()

    # Send request to the endpoint
    response = client.get(
        '/mattermost/search/get',
        headers={},
        params={
            'team_id': mock_channel.team_id,  # Use the valid team_id from the mock channel
            'search_terms': ['substring'],   # Valid search terms
        }
    )

    # Assert the response status code
    assert response.status_code == 200


def test_create_conversation_objects_with_none_document():
    """Test that create_conversation_objects raises HTTPException when document is None"""
    from app.mattermost.router import create_conversation_objects
    from app.ppg_common.schemas.mattermost.mattermost_documents import ThreadTypeEnum, InfoTypeEnum
    
    mock_db = MagicMock(spec=Session)
    
    # Create a mock mm_document_obj with proper UUID types
    mock_mm_doc = MagicMock()
    mock_mm_doc.message_id = "test-message-id"
    mock_mm_doc.root_message_id = "test-root-id"
    mock_mm_doc.type = "test-type"
    mock_mm_doc.channel = uuid.uuid4()
    mock_mm_doc.user = uuid.uuid4()
    mock_mm_doc.document = uuid.uuid4()
    mock_mm_doc.info_type = InfoTypeEnum.CHAT
    
    # Mock the crud methods
    with patch('app.mattermost.router.crud_mattermost') as mock_crud_mm:
        with patch('app.mattermost.router.crud_document') as mock_crud_doc:
            # Return the mock mm_document_obj
            mock_crud_mm.mattermost_documents.get_by_message_id.return_value = mock_mm_doc
            
            # Return None for document.get to trigger the check
            mock_crud_doc.document.get.return_value = None
            
            # Create test dataframe
            test_df = pd.DataFrame([{
                'message_id': 'test-message-id',
                'document_id': 'test-doc-id',
                'thread': 'test thread',
                'hashtags': '',
                'has_reactions': False,
                'props': {},
                'metadata': {}
            }])
            
            # Should raise HTTPException
            with pytest.raises(Exception):
                create_conversation_objects(mock_db, ThreadTypeEnum.THREAD, test_df)


def test_create_conversation_objects_converts_attributes_to_strings():
    """Test that create_conversation_objects properly converts model attributes to strings"""
    from app.mattermost.router import create_conversation_objects
    from app.ppg_common.schemas.mattermost.mattermost_documents import ThreadTypeEnum, InfoTypeEnum
    
    mock_db = MagicMock(spec=Session)
    
    # Create mock objects with proper UUID types
    mock_mm_doc = MagicMock()
    mock_mm_doc.message_id = "test-message-id"
    mock_mm_doc.root_message_id = "test-root-id"
    mock_mm_doc.type = "test-type"
    mock_mm_doc.channel = uuid.uuid4()
    mock_mm_doc.user = uuid.uuid4()
    mock_mm_doc.document = uuid.uuid4()
    mock_mm_doc.info_type = InfoTypeEnum.CHAT
    
    mock_document = MagicMock()
    mock_document.id = "doc-id"
    mock_document.original_created_time = datetime.datetime.now()
    
    mock_updated_doc = MagicMock()
    
    with patch('app.mattermost.router.crud_mattermost') as mock_crud_mm:
        with patch('app.mattermost.router.crud_document') as mock_crud_doc:
            mock_crud_mm.mattermost_documents.get_by_message_id.return_value = mock_mm_doc
            mock_crud_doc.document.get.return_value = mock_document
            mock_crud_doc.document.update.return_value = mock_document
            mock_crud_mm.mattermost_documents.update.return_value = mock_updated_doc
            
            test_df = pd.DataFrame([{
                'message_id': 'test-message-id',
                'document_id': 'test-doc-id',
                'thread': 'test thread',
                'hashtags': 'tag1 tag2',
                'has_reactions': True,
                'props': {'key': 'value'},
                'metadata': {'meta': 'data'}
            }])
            
            result = create_conversation_objects(mock_db, ThreadTypeEnum.THREAD, test_df)
            
            # Verify update was called
            assert mock_crud_mm.mattermost_documents.update.called
            
            # Get the call arguments
            call_args = mock_crud_mm.mattermost_documents.update.call_args
            obj_in = call_args.kwargs['obj_in']
            
            # Verify that string conversions were applied
            assert isinstance(obj_in.message_id, str)
            assert isinstance(obj_in.root_message_id, str)
            assert isinstance(obj_in.type, str)


def test_create_conversation_objects_handles_new_documents():
    """Test that create_conversation_objects creates new documents when they don't exist"""
    from app.mattermost.router import create_conversation_objects
    from app.ppg_common.schemas.mattermost.mattermost_documents import ThreadTypeEnum
    
    mock_db = MagicMock(spec=Session)
    
    with patch('app.mattermost.router.crud_mattermost') as mock_crud_mm:
        # Return None to indicate document doesn't exist
        mock_crud_mm.mattermost_documents.get_by_message_id.return_value = None
        
        # Mock create_all_using_df to return list of created documents
        mock_new_docs = [MagicMock()]
        mock_crud_mm.mattermost_documents.create_all_using_df.return_value = mock_new_docs
        
        test_df = pd.DataFrame([{
            'message_id': 'new-message-id',
            'document_id': 'new-doc-id',
            'thread': 'test thread',
            'hashtags': '',
            'has_reactions': False,
            'props': {},
            'metadata': {}
        }])
        
        result = create_conversation_objects(mock_db, ThreadTypeEnum.THREAD, test_df)
        
        # Verify create_all_using_df was called
        mock_crud_mm.mattermost_documents.create_all_using_df.assert_called_once()
        
        # Verify result contains the new document
        assert len(result) == 1
        assert result[0] == mock_new_docs[0]
