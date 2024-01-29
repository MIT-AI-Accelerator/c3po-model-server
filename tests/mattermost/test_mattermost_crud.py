import uuid
import pandas as pd
from sqlalchemy.orm import Session
from app.core.config import environment_settings
from app.mattermost.models.mattermost_documents import MattermostDocumentModel
import app.mattermost.crud.crud_mattermost as crud
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.crud import crud_document


def test_crud_mattermost(db: Session):
    # test crud mattermost

    # test mattermost channel
    channel_info = dict({'id': str(uuid.uuid4()), 'name': 'my channel', 'team_id': str(
        uuid.uuid4()), 'team_name': 'my team'})
    channel_db_obj = crud.populate_mm_channel_info(
        db, channel_info=channel_info)

    assert channel_db_obj.channel_id == channel_info['id']
    assert channel_db_obj.channel_name == channel_info['name']
    assert channel_db_obj.team_id == channel_info['team_id']
    assert channel_db_obj.team_name == channel_info['team_name']
    assert crud.mattermost_channels.get_by_channel_id(
        db, channel_id=channel_info['id']) is not None
    assert crud.mattermost_channels.get_by_channel_name(
        db, team_name=channel_info['team_name'], channel_name=channel_info['name']) is not None
    assert channel_info['id'] in crud.mattermost_channels.get_all_channel_ids(
        db)

    # test mattermost user
    user = str(uuid.uuid4())
    mm_user = dict({'id': user, 'username': user})
    teams = dict({'0': 'a team', '1': 'another team'})
    user_db_obj = crud.populate_mm_user_info(db, mm_user=mm_user, teams=teams)

    assert user_db_obj.user_id == mm_user['id']
    assert user_db_obj.user_name == mm_user['username']
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
        db, message_id=obj_in.message_id) is not None
    assert crud.mattermost_documents.get_all_channel_documents(
        db, channels=[obj_in.channel]) is not None
    assert not crud.mattermost_documents.get_document_dataframe(
        db, document_uuids=[db_obj.id]).empty


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
    mock_user_data = (dict({'id': user, 'username': user}), pd.DataFrame())
    mocker.patch(
        'ppg.mattermost_utils.get_user_info', return_value=mock_user_data)

    mock_team_data = pd.DataFrame()
    mocker.patch('ppg.mattermost_utils.get_team_channels', return_value=mock_team_data)
    mocker.patch('ppg.mattermost_utils.get_all_user_channels', return_value=mock_team_data)

    user_obj = crud.populate_mm_user_team_info(db, user_name=user)

    assert user_obj.user_name == user


def test_convert_conversation_threads():

    msg1 = 'message 1.'
    msg2 = 'message 2.'

    # construct message data frame with reply and convert to conversation thread
    document_df = pd.DataFrame()
    document_df = pd.concat([document_df,  pd.DataFrame(
        [{'id': '1', 'message': msg1, 'root_id': ''}])])
    document_df = pd.concat([document_df,  pd.DataFrame(
        [{'id': '2', 'message': msg2, 'root_id': '1'}])])

    conversation_df = crud.convert_conversation_threads(document_df)

    assert len(conversation_df) == (len(document_df) - 1)
    assert conversation_df['message'].iloc[0] == '%s\n%s' % (msg1, msg2)
