import uuid
from sqlalchemy.orm import Session
from app.mattermost.models.mattermost_channels import MattermostChannelModel
from app.mattermost.models.mattermost_users import MattermostUserModel
from app.mattermost.models.mattermost_documents import MattermostDocumentModel
import app.mattermost.crud.crud_mattermost as crud
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.crud import crud_document


# test crud mattermost
def test_crud_mattermost(db: Session):

    # test mattermost channel
    channel_obj_in = MattermostChannelModel(
        channel_id=str(uuid.uuid4()),
        channel_name='my channel',
        team_id=str(uuid.uuid4()),
        team_name='my team'
    )

    channel_db_obj = crud.mattermost_channels.create(db, obj_in=channel_obj_in)

    assert channel_db_obj.channel_id == channel_obj_in.channel_id
    assert channel_db_obj.channel_name == channel_obj_in.channel_name
    assert channel_db_obj.team_id == channel_obj_in.team_id
    assert channel_db_obj.team_name == channel_obj_in.team_name

    # test mattermost user
    user = str(uuid.uuid4())
    user_obj_in = MattermostUserModel(
        user_id=user,
        user_name=user,
        teams=dict({'0': 'a team', '1': 'another team'})
    )

    user_db_obj = crud.mattermost_users.create(db, obj_in=user_obj_in)

    assert user_db_obj.user_id == user_obj_in.user_id
    assert user_db_obj.user_name == user_obj_in.user_name
    assert user_db_obj.teams == user_obj_in.teams

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
