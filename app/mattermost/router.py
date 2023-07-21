"""mattermost router"""
from fastapi import APIRouter
from app.core.config import settings
import json 
import pandas as pd
from typing import Union
from tqdm import tqdm
from fastapi import Depends
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.aimodels.bertopic.schemas.document import DocumentCreate
from app.aimodels.bertopic import crud as crud_document
from app.mattermost.services import mattermost_utils
from app.mattermost.crud import crud_mattermost
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

"""populate the db with mattermost user info"""
@router.post("/mattermost/user/upload", summary="Mattermost User Info", response_description="")
async def upload_mm_user_info(user_name: str, db: Session = Depends(get_db)):
    """
    Upload Mattermost user info.

    - **user_name**: Required.  Mattermost user name.
    """

    return crud_mattermost.populate_mm_user_info(db, user_name=user_name)

"""get a list of channels by mattermost user name"""
@router.get("/mattermost/user/get", summary="Mattermost User Info", response_description="")
async def get_mm_user_info(user_name: str, db: Session = Depends(get_db)):
    """
    Get Mattermost user info.

    - **user_name**: Required.  Mattermost user name.
    """

    return crud_mattermost.mattermost_users.get_by_user_name(db, user_name=user_name)

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
        df = mattermost_utils.get_channel_posts(settings.mm_base_url, settings.mm_token, channel_id).assign(channel=channel_obj.id)
        adf = pd.concat([adf, df], ignore_index=True)
    channel_uuids = adf['channel'].unique()

    user_ids = adf['user_id'].unique()
    for uid in tqdm(user_ids):
        user_obj = crud_mattermost.mattermost_users.get_by_user_id(db, user_id=uid)
        if user_obj is None:
            user_name = mattermost_utils.get_user_name(settings.mm_base_url, settings.mm_token, uid)
            user_obj = crud_mattermost.populate_mm_user_info(db, user_name=user_name)
        adf.loc[adf['user_id'] == uid, 'user'] = user_obj.id

    channel_document_objs = crud_mattermost.mattermost_documents.get_all_channel_documents(db, channels=channel_uuids)
    existing_ids = [obj.message_id for obj in channel_document_objs]
    adf = adf[~adf.id.isin(existing_ids)].drop_duplicates(subset='id')

    # note: database objects are not returned in the same order as input!
    # do not use the following and try to align doc IDs with mattermost docs
    # document_objs = crud_document.document.create_all_using_id(db, obj_in_list=documents)
    mattermost_documents = []
    for key, row in tqdm(adf.iterrows()):
        document = DocumentCreate(
            text=row['message'],
            original_created_time = row['create_at'])
        document_obj = crud_document.document.create(db, obj_in=document)

        mattermost_documents = mattermost_documents + [MattermostDocumentCreate(
            message_id = row['id'],
            channel = row['channel'],
            user = row['user'], 
            document = document_obj.id)]

    return channel_document_objs + crud_mattermost.mattermost_documents.create_all_using_id(db, obj_in_list=mattermost_documents)

"""retrieve mattermost posts by for a channel"""
@router.get("/mattermost/documents/get", summary="Mattermost Documents", response_description="")
async def get_mm_channel_docs(team_name: str, channel_name: str, db: Session = Depends(get_db)):
    """
    Get Mattermost documents.

    - **team_name**: Required.  Mattermost team to query for posts.
    - **channel_name**: Required.  Mattermost channel to query for posts.
    """

    # TODO filter on time

    channel = crud_mattermost.mattermost_channels.get_by_channel_name(db, team_name=team_name, channel_name=channel_name)
    return crud_mattermost.mattermost_documents.get_all_channel_documents(db, channels=[channel.id])
