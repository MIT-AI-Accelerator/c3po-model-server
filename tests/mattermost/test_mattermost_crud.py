import uuid
import pandas as pd
from sqlalchemy.orm import Session
from app.core.config import environment_settings, settings
from app.mattermost.models.mattermost_documents import MattermostDocumentModel
import app.mattermost.crud.crud_mattermost as crud
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.crud import crud_document


def test_crud_mattermost(db: Session, caplog):
    # test crud mattermost

    # test mattermost channel
    channel_info = dict({'id': str(uuid.uuid4()),
                         'name': 'my channel',
                         'team_id': str(uuid.uuid4()),
                         'team_name': 'my team',
                         'display_name': 'my channel',
                         'type': 'P',
                         'header': 'my header',
                         'purpose': 'my purpose'})
    channel_db_obj = crud.populate_mm_channel_info(
        db, channel_info=channel_info)

    assert channel_db_obj.channel_id == channel_info['id']
    assert channel_db_obj.channel_name == channel_info['name']
    assert channel_db_obj.team_id == channel_info['team_id']
    assert channel_db_obj.team_name == channel_info['team_name']
    assert channel_db_obj.display_name == channel_info['display_name']
    assert channel_db_obj.type == channel_info['type']
    assert channel_db_obj.header == channel_info['header']
    assert channel_db_obj.purpose == channel_info['purpose']
    assert crud.mattermost_channels.get_by_channel_id(
        db, channel_id=channel_info['id']) is not None
    assert crud.mattermost_channels.get_by_channel_name(
        db, team_name=channel_info['team_name'], channel_name=channel_info['name']) is not None
    assert channel_info['id'] in crud.mattermost_channels.get_all_channel_ids(
        db)

    # test mattermost user
    user = str(uuid.uuid4())
    mm_user = dict({'id': user,
                    'username': user,
                    'nickname': user,
                    'first_name': 'afirstname',
                    'last_name': 'alastname',
                    'position': 'something important',
                    'email': '%s@nitmre.mil' % user})
    teams = dict({'0': 'a team', '1': 'another team'})
    user_db_obj = crud.populate_mm_user_info(db, mm_user=mm_user, teams=teams)

    assert user_db_obj.user_id == mm_user['id']
    assert user_db_obj.user_name == mm_user['username']
    assert user_db_obj.nickname == mm_user['nickname']
    assert user_db_obj.first_name == mm_user['first_name']
    assert user_db_obj.last_name == mm_user['last_name']
    assert user_db_obj.position == mm_user['position']
    assert user_db_obj.email == mm_user['email']
    assert user_db_obj.teams == teams
    assert crud.mattermost_users.get_by_user_id(
        db, user_id=mm_user['id']) is not None
    assert crud.mattermost_users.get_by_user_name(
        db, user_name=mm_user['username']) is not None

    # test mattermost document
    doc_db_obj = crud_document.document.create(db,
                                               obj_in=DocumentModel(text='my document'))

    obj_in = MattermostDocumentModel(
        message_id=str(uuid.uuid4()),
        root_message_id=str(uuid.uuid4()),
        channel=channel_db_obj.id,
        user=user_db_obj.id,
        document=doc_db_obj.id
    )

    db_obj = crud.mattermost_documents.create(db, obj_in=obj_in)

    assert db_obj.message_id == obj_in.message_id
    assert db_obj.root_message_id == obj_in.root_message_id
    assert db_obj.channel == obj_in.channel
    assert db_obj.user == obj_in.user
    assert crud.mattermost_documents.get_by_message_id(
        db, message_id='') is None
    assert crud.mattermost_documents.get_by_message_id(
        db, message_id=obj_in.message_id) is not None
    assert crud.mattermost_documents.get_all_by_message_id(
        db, message_id='') is None
    assert crud.mattermost_documents.get_all_by_message_id(
        db, message_id=obj_in.message_id) is not None
    assert crud.mattermost_documents.get_all_channel_documents(
        db, channels=[obj_in.channel]) is not None
    assert crud.mattermost_documents.get_all_channel_documents(
        db, channels=[obj_in.channel], history_depth=45) is not None
    assert not crud.mattermost_documents.get_mm_document_dataframe(
        db, mm_document_uuids=[db_obj.id]).empty

    ddf = crud.mattermost_documents.get_document_dataframe(db, document_uuids=[doc_db_obj.id])
    mmdf = ddf[ddf.document_uuid == doc_db_obj.id]
    assert mmdf.loc[0, 'document_uuid'] == doc_db_obj.id
    assert mmdf.loc[0, 'mm_doc_uuid'] == db_obj.id
    assert mmdf.loc[0, 'message_id'] == obj_in.message_id
    assert mmdf.loc[0, 'message'] == 'my document'
    assert mmdf.loc[0, 'root_id'] == obj_in.root_message_id
    assert mmdf.loc[0, 'type'] == obj_in.type
    assert mmdf.loc[0, 'user_uuid'] == user_db_obj.id
    assert mmdf.loc[0, 'user_id'] == mm_user['id']
    assert mmdf.loc[0, 'user_name'] == mm_user['username']
    assert mmdf.loc[0, 'nickname'] == mm_user['nickname']
    assert mmdf.loc[0, 'channel_uuid'] == channel_db_obj.id
    assert mmdf.loc[0, 'channel_name'] == channel_info['name']
    assert mmdf.loc[0, 'team_name'] == channel_info['team_name']
    assert mmdf.loc[0, 'mm_link'] == '%s/%s/pl/%s' % (
        settings.mm_base_url, channel_info['team_name'], obj_in.message_id)
    assert mmdf.loc[0, 'create_at'] == doc_db_obj.original_created_time




def test_populate_mm_user_team_info_local(db: Session):
    # test user info in database

    if environment_settings.environment == 'test':
        return

    mm_name = 'nitmre-bot'
    user_obj = crud.populate_mm_user_team_info(db, user_name=mm_name)

    assert user_obj.user_name == mm_name


def test_populate_mm_user_team_info(db: Session, mocker):
    # test user info in database

    # mock mattermost_utils for staging pipeline
    user = str(uuid.uuid4())
    mock_user_data = (dict({'id': user,
                            'username': user,
                            'nickname': user,
                            'first_name': 'afirstname',
                            'last_name': 'alastname',
                            'position': 'something important',
                            'email': '%s@nitmre.mil' % user}),
                      pd.DataFrame())
    mocker.patch(
        'ppg.services.mattermost_utils.get_user_info', return_value=mock_user_data)

    mock_team_data = pd.DataFrame()
    mocker.patch('ppg.services.mattermost_utils.get_user_team_channels', return_value=mock_team_data)
    mocker.patch('ppg.services.mattermost_utils.get_all_user_team_channels', return_value=mock_team_data)

    user_obj = crud.populate_mm_user_team_info(db, user_name=user)

    assert user_obj.user_name == user


def test_convert_conversation_threads():

    msg1 = 'message 1.'
    msg2 = 'message 2.'

    # construct message data frame with reply and convert to conversation thread
    document_df = pd.DataFrame()
    document_df = pd.concat([document_df,  pd.DataFrame(
        [{'message_id': '1', 'message': msg1, 'root_id': ''}])])
    document_df = pd.concat([document_df,  pd.DataFrame(
        [{'message_id': '2', 'message': msg2, 'root_id': '1'}])])

    conversation_df = crud.convert_conversation_threads(document_df)

    assert len(conversation_df) == (len(document_df) - 1)
    assert conversation_df['message'].iloc[0] == '%s\n%s' % (msg1, msg2)
