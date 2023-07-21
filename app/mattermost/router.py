"""mattermost router"""
from fastapi import APIRouter
from app.core.config import settings
import json 
import pandas as pd
from fastapi import Depends
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.aimodels.bertopic.schemas.document import DocumentCreate
from app.aimodels.bertopic import crud as crud_document
import app.mattermost.services.mattermost_utils as mm
from .crud import crud_mattermost 
from .schemas.mattermost_documents import MattermostDocumentCreate
from .models.mattermost_documents import MattermostDocumentModel
from .schemas.mattermost_users import MattermostUserCreate, MattermostUserUpdate
from .models.mattermost_users import MattermostUserModel
from .schemas.mattermost_channels import MattermostChannelCreate
from .models.mattermost_channels import MattermostChannelModel

"""mattermost section"""
router = APIRouter(
    prefix="", tags=["Mattermost"]
)
    
def populate_mm_user_info(user_name: str, db: Session) -> MattermostUserModel:

    # add or update user info in db 
    (mm_user, tdf) = mm.get_user_info(settings.mm_base_url, settings.mm_token, user_name)
    teams = dict()
    if not tdf.empty:
        teams = tdf['name'].to_dict()

    user_obj = crud_mattermost.mattermost_users.get_by_user_name(db, user_name=user_name)
    if user_obj is None:
        create_user = MattermostUserCreate(user_id=mm_user['id'], user_name=mm_user['username'], teams=teams)
        user_obj = crud_mattermost.mattermost_users.create(db, obj_in=create_user)
        # todo report error
    else:
        update_user = MattermostUserUpdate(user_id= mm_user['id'], user_name=mm_user['username'], teams=teams)
        user_obj = crud_mattermost.mattermost_users.update(db, db_obj=user_obj, obj_in=update_user)
        # todo report error
        
    # retrieve existing channels from db
    cdf = mm.get_all_user_channels(settings.mm_base_url, settings.mm_token, mm_user['id'], teams)
    if not cdf.empty:
        existing_ids = crud_mattermost.mattermost_channels.get_all_channel_ids(db)
        cdf = cdf[~cdf.id.isin(existing_ids)]

        # add new channels to db
        channels = [MattermostChannelCreate(
            channel_id = row['id'],
            channel_name = row['name'],
            team_id = row['team_id'],
            team_name = row['team_name']
        ) for key, row in cdf.iterrows()]
        new_channels = crud_mattermost.mattermost_channels.create_all_using_id(db, obj_in_list=channels)
        # TODO report error

    return user_obj

"""get a list of channels by mattermost user name"""
@router.get("/mattermost/user-info/", summary="Mattermost User Info", response_description="")
async def get_mm_user_info(user_name: str, db: Session = Depends(get_db)):
    """
    Get Mattermost user info.

    - **user_name**: Required.  Mattermost user name.
    """
    return populate_mm_user_info(user_name, db)


"""populate the db with mattermost posts from a list of channel ids"""
@router.post("/mattermost/documents/upload", summary="Mattermost Documents", response_description="")
async def upload_mm_channel_docs(channel_ids: list[str], db: Session = Depends(get_db)):
    """
    Upload Mattermost documents.

    - **channel_ids**: Required.  Mattermost channels to query for posts.
    """

    adf = pd.DataFrame()
    for channel_id in channel_ids:
        channel_obj = crud_mattermost.mattermost_channels.get_by_channel_id(db, channel_id=channel_id)
        # if channel_obj is not None: TODO add channel info to db
        # else:
        df = mm.get_channel_posts(settings.mm_base_url, settings.mm_token, channel_id).assign(channel=channel_obj.id)
        adf = pd.concat([adf, df], ignore_index=True)
    channel_uuids = adf['channel'].unique()

    user_ids = adf['user_id'].unique()
    for uid in user_ids:
        user_obj = crud_mattermost.mattermost_users.get_by_user_id(db, user_id=uid)
        if user_obj is None:
            user_name = mm.get_user_name(settings.mm_base_url, settings.mm_token, uid)
            user_obj = populate_mm_user_info(user_name, db)
        adf.loc[adf['user_id'] == uid, 'user'] = user_obj.id

    channel_document_objs = crud_mattermost.mattermost_documents.get_all_channel_documents(db, channels=channel_uuids)
    existing_ids = [obj.message_id for obj in channel_document_objs]
    adf = adf[~adf.id.isin(existing_ids)]

    if not adf.empty:
        documents = [DocumentCreate(
            text = row['message'],
            original_created_time = row['create_at']
        ) for key, row in adf.iterrows()]
        document_objs = crud_document.document.create_all_using_id(db, obj_in_list=documents)
        adf = adf.assign(document=[obj.id for obj in document_objs])

        # TODO load all users in init
        # TODO need endpoints for get vs update user, get vs create documents
        # TODO filter on time

        mattermost_document = [MattermostDocumentCreate(
            message_id = row['id'],
            channel = row['channel'],
            user = row['user'], 
            document = row['document']
        ) for key, row in adf.iterrows()]
        channel_document_objs = channel_document_objs + crud_mattermost.mattermost_documents.create_all_using_id(db, obj_in_list=mattermost_document)

    return channel_document_objs

"""retrieve mattermost posts by for a channel"""
@router.get("/mattermost/documents/get", summary="Mattermost Documents", response_description="")
async def get_mm_channel_docs(team_name: str, channel_name: str, db: Session = Depends(get_db)):
    """
    Get Mattermost documents.

    - **team_name**: Required.  Mattermost team to query for posts.
    - **channel_name**: Required.  Mattermost channel to query for posts.
    """
    channel = crud_mattermost.mattermost_channels.get_by_channel_name(db, team_name=team_name, channel_name=channel_name)
    return crud_mattermost.mattermost_documents.get_all_channel_documents(db, channels=[channel.id])
