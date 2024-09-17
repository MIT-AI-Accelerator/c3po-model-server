import uuid, datetime
import pandas as pd
import app.mattermost.crud.crud_mattermost as crud
from _pytest.monkeypatch import MonkeyPatch
from sqlalchemy.orm import Session
from app.core.config import environment_settings, settings
from app.mattermost.models.mattermost_documents import MattermostDocumentModel
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.crud import crud_document
from ppg.core.config import OriginationEnum
from ppg.schemas.mattermost.mattermost_documents import InfoTypeEnum


def test_crud_mattermost(db: Session):
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
    assert crud.mattermost_channels.get_by_channel_id(
        db, channel_id='') is None
    assert crud.mattermost_channels.get_by_channel_name(
        db, team_name=channel_info['team_name'], channel_name=channel_info['name']) is not None
    assert crud.mattermost_channels.get_by_channel_name(
        db, team_name='', channel_name=channel_info['name']) is None
    assert crud.mattermost_channels.get_by_channel_name(
        db, team_name=channel_info['team_name'], channel_name='') is None
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
    assert crud.mattermost_users.get_by_user_id(
        db, user_id='') is None
    assert crud.mattermost_users.get_by_user_name(
        db, user_name=mm_user['username']) is not None
    assert crud.mattermost_users.get_by_user_name(
        db, user_name='') is None

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

    # get document as a dataframe
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

    # create a new mattermost document with same channel and user
    cdf = pd.DataFrame([{'message_id': str(uuid.uuid4()),
                         'root_id': str(uuid.uuid4()),
                         'channel': channel_db_obj.id,
                         'user': user_db_obj.id,
                         'message': 'create with df',
                         'create_at': datetime.datetime.now(),
                         'type': 'C',
                         'hashtags': 'eggo',
                         'has_reactions': False,
                         'props': {'leggo': 'myeggo'},
                         'metadata': {'cuckoo': 'forcocoapuffs'},
                         }])
    cdf_is_thread = cdf.loc[0, 'root_id'] == db_obj.root_message_id
    mmdocs = crud.mattermost_documents.create_all_using_df(db, ddf=cdf, is_thread=cdf_is_thread)
    assert len(mmdocs) == 1
    mmdoc = mmdocs[0]
    newdoc = crud_document.document.get(db, mmdoc.document)
    assert mmdoc.message_id == cdf.loc[0, 'message_id']
    assert mmdoc.root_message_id == cdf.loc[0, 'root_id']
    assert mmdoc.channel == cdf.loc[0, 'channel']
    assert mmdoc.user == cdf.loc[0, 'user']
    assert newdoc.text == cdf.loc[0, 'message']
    assert mmdoc.type == cdf.loc[0, 'type']
    assert mmdoc.hashtags == cdf.loc[0, 'hashtags']
    assert mmdoc.has_reactions == cdf.loc[0, 'has_reactions']
    assert mmdoc.props == cdf.loc[0, 'props']
    assert mmdoc.doc_metadata == cdf.loc[0, 'metadata']
    assert mmdoc.is_thread == cdf_is_thread
    assert mmdoc.originated_from == settings.originated_from


def test_populate_mm_user_team_info_local(db: Session, monkeypatch: MonkeyPatch):
    # test user info in database

    if environment_settings.environment == 'test':
        return

    # see note at tests/mattermost/test_mattermost_router.py::test_get_mattermost_user_info
    monkeypatch.setattr(settings, 'originated_from', OriginationEnum.ORIGINATED_FROM_APP)

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
        [{'message_id': '1', 'message': msg1, 'root_id': '', 'info_type': InfoTypeEnum.CHAT}])])
    document_df = pd.concat([document_df,  pd.DataFrame(
        [{'message_id': '2', 'message': msg2, 'root_id': '1', 'info_type': InfoTypeEnum.CHAT}])])

    conversation_df = crud.convert_conversation_threads(document_df)

    assert len(conversation_df) == (len(document_df) - 1)
    assert conversation_df['message'].iloc[0] == '%s\n%s' % (msg1, msg2)


def test_parse_props():
    # test parse mm props

    ittl = 'DIPS'
    aname = ''
    imsg = 'Inspiring Cat Overcomes Prejudice To Win Westminster Dog Show.'
    jobj = {'attachments': [{'id': 0,
                             'author_name': aname,
                             'title': ittl,
                             'text': imsg,
                             'fallback': '',
                             'fields': []}]}
    itype, omsg = crud.parse_props(jobj)
    emsg = '[%s] %s' % (ittl, imsg)

    assert itype == InfoTypeEnum.ENVISION
    assert omsg[0:len(emsg)] == emsg

    ittl = ''
    aname = 'CAMPS'
    jobj['attachments'][0]['title'] = ittl
    jobj['attachments'][0]['author_name'] = aname
    itype, omsg = crud.parse_props(jobj)
    emsg = '[%s] %s' % (ittl, imsg)

    assert itype == InfoTypeEnum.CAMPS
    assert omsg[0:len(emsg)] == emsg

    ittl = ''
    aname = 'ARINC'
    jobj['attachments'][0]['title'] = ittl
    jobj['attachments'][0]['author_name'] = aname
    itype, omsg = crud.parse_props(jobj)
    emsg = '[%s] %s' % (ittl, imsg)

    assert itype == InfoTypeEnum.ARINC
    assert omsg[0:len(emsg)] == emsg

    ittl = ''
    aname = ''
    jobj['attachments'][0]['title'] = ittl
    jobj['attachments'][0]['author_name'] = aname
    itype, omsg = crud.parse_props(jobj)
    emsg = '[%s] %s' % (ittl, imsg)

    assert itype == InfoTypeEnum.UDL
    assert omsg[0:len(emsg)] == emsg

def test_parse_props_notam():
    # test parse mm notam props

    ittl = 'NOTAM'
    imsg = 'House Cat Announces Plans To Just Sit There For 46 Minutes.'
    jobj = {'attachments': [{'id': 0,
                             'author_name': '',
                             'title': ittl,
                             'text': imsg,
                             'fallback': '',
                             'fields': [{'title': 'Location', 'value': 'KCAT', 'short': True},
                                        {'title': 'Valid', 'value': '4149/0409Z - 4201/2359Z', 'short': True}]}]}
    itype, omsg = crud.parse_props(jobj)
    emsg = '[%s] %s' % (ittl, imsg)

    assert itype == InfoTypeEnum.NOTAM
    assert omsg[0:len(emsg)] == emsg


def test_parse_props_acars():
    # test parse mm acars props

    ittl = 'ACARS'
    imsg = '8th Cat Acquired In Hopes Of Easing Tension Between First 7 Cats.'
    jobj = {'attachments': [{'id': 0,
                             'author_name': '',
                             'title': ittl,
                             'text': imsg,
                             'fallback': '',
                             'fields': [{'title': 'Tail #', 'value': '8675309', 'short': True},
                                        {'title': 'Mission #', 'value': '8675309', 'short': True},
                                        {'title': 'Callsign', 'value': 'CAT123', 'short': True}]}]}
    itype, omsg = crud.parse_props(jobj)
    emsg = '[%s] %s' % (ittl, imsg)

    assert itype == InfoTypeEnum.ACARS
    assert omsg[0:len(emsg)] == emsg


def test_parse_props_dataminr():
    # test parse mm dataminr props

    imsg = 'War On String May Be Unwinnable, Says Cat General.'
    jobj = {'attachments': [{'id': 0,
                             'author_name': 'Dataminr',
                             'title': imsg,
                             'text': '',
                             'fallback': '',
                             'fields': [{'title': 'Alert Type', 'value': 'Urgent', 'short': False},
                                        {'title': 'Event Time', 'value': '26/06/2024 18:08:19', 'short': False},
                                        {'title': 'Event Location', 'value': 'Lexington, MA USA\n', 'short': False},
                                        {'title': 'Nearby Airfields', 'value': 'KCAT\n', 'short': False}]}]}
    itype, omsg = crud.parse_props(jobj)

    assert itype == InfoTypeEnum.DATAMINR
    assert omsg[0:len(imsg)] == imsg
