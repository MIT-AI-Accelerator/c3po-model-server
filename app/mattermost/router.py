"""mattermost router"""
from typing import Union
from pydantic import BaseModel
from fastapi import Depends, APIRouter, HTTPException
import pandas as pd
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.errors import HTTPValidationError
from app.dependencies import get_db
from app.aimodels.bertopic.schemas.document import DocumentCreate
from app.aimodels.bertopic import crud as crud_document
from app.mattermost.services import mattermost_utils
from app.mattermost.crud import crud_mattermost
from .schemas.mattermost_documents import MattermostDocument, MattermostDocumentCreate
from .schemas.mattermost_users import MattermostUser

"""mattermost section"""
router = APIRouter(
    prefix="", tags=["Mattermost"]
)

class UploadUserRequest(BaseModel):
    user_name: str

@router.post(
    "/mattermost/user/upload",
    response_model=Union[MattermostUser, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Upload Mattermost user info",
    response_description="Uploaded Mattermost user info")
async def upload_mm_user_info(request: UploadUserRequest, db: Session = Depends(get_db)) -> (
    Union[MattermostUser, HTTPValidationError]
):
    """
    Populate the db with mattermost user info

    - **user_name**: Required.  Mattermost user name.
    """

    user_obj = crud_mattermost.populate_mm_user_info(db, user_name=request.user_name)
    if not user_obj:
        raise HTTPException(
            status_code=422, detail="Mattermost user not found")

    return user_obj


@router.get(
    "/mattermost/user/get",
    response_model=Union[MattermostUser, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Get Mattermost user info",
    response_description="Retrieved Mattermost user info")
async def get_mm_user_info(user_name: str, db: Session = Depends(get_db)) -> (
    Union[MattermostUser, HTTPValidationError]
):
    """
    Get a list of channels by mattermost user name

    - **user_name**: Required.  Mattermost user name.
    """

    user_obj = crud_mattermost.mattermost_users.get_by_user_name(
        db, user_name=user_name)
    if not user_obj:
        raise HTTPException(
            status_code=422, detail="Mattermost user not found")

    return user_obj

class UploadDocumentRequest(BaseModel):
    channel_ids: list[str] = []

@router.post(
    "/mattermost/documents/upload",
    response_model=Union[list[MattermostDocument], HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Upload Mattermost documents",
    response_description="Uploaded Mattermost documents")
async def upload_mm_channel_docs(request: UploadDocumentRequest, db: Session = Depends(get_db)) -> (
    Union[list[MattermostDocument], HTTPValidationError]
):
    """
    Populate the db with mattermost posts from a list of channel ids

    - **channel_ids**: Required.  Mattermost channels to query for posts.
    """

    if not request.channel_ids:
        raise HTTPException(status_code=422, detail="No Mattermost channels requested")

    adf = pd.DataFrame()
    for channel_id in request.channel_ids:
        channel_obj = crud_mattermost.mattermost_channels.get_by_channel_id(
            db, channel_id=channel_id)
        if not channel_obj:
            channel_info = mattermost_utils.get_channel_info(
                settings.mm_base_url, settings.mm_token, channel_id)
            if not channel_info:
                raise HTTPException(
                    status_code=422, detail="Mattermost channel not found")
            channel_obj = crud_mattermost.populate_mm_channel_info(
                db, channel_info=channel_info)
        df = mattermost_utils.get_channel_posts(
            settings.mm_base_url, settings.mm_token, channel_id).assign(channel=channel_obj.id)
        adf = pd.concat([adf, df], ignore_index=True)
    channel_uuids = adf['channel'].unique()

    user_ids = adf['user_id'].unique()
    for uid in user_ids:
        user_obj = crud_mattermost.mattermost_users.get_by_user_id(
            db, user_id=uid)
        if user_obj is None:
            user_name = mattermost_utils.get_user_name(
                settings.mm_base_url, settings.mm_token, uid)
            user_obj = crud_mattermost.populate_mm_user_info(
                db, user_name=user_name)
        adf.loc[adf['user_id'] == uid, 'user'] = user_obj.id

    channel_document_objs = crud_mattermost.mattermost_documents.get_all_channel_documents(
        db, channels=channel_uuids)
    existing_ids = [obj.message_id for obj in channel_document_objs]
    adf = adf[~adf.id.isin(existing_ids)].drop_duplicates(subset='id')

    # note: database objects are not returned in the same order as input!
    # do not use the following and try to align doc IDs with mattermost docs
    # document_objs = crud_document.document.create_all_using_id(db, obj_in_list=documents)
    mattermost_documents = []
    for key, row in adf.iterrows():
        document = DocumentCreate(
            text=row['message'],
            original_created_time=row['create_at'])
        document_obj = crud_document.document.create(db, obj_in=document)

        mattermost_documents = mattermost_documents + [MattermostDocumentCreate(
            message_id=row['id'],
            root_message_id=row['root_id'],
            channel=row['channel'],
            user=row['user'],
            document=document_obj.id)]

    return channel_document_objs + crud_mattermost.mattermost_documents.create_all_using_id(db, obj_in_list=mattermost_documents)


@router.get(
    "/mattermost/documents/get",
    response_model=Union[list[MattermostDocument], HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve Mattermost documents",
    response_description="Retrieved Mattermost documents")
async def get_mm_channel_docs(team_name: str, channel_name: str, db: Session = Depends(get_db)) -> (
    Union[list[MattermostDocument], HTTPValidationError]
):
    """
    Retrieve mattermost posts by for a channel

    - **team_name**: Required.  Mattermost team to query for posts.
    - **channel_name**: Required.  Mattermost channel to query for posts.
    """

    channel_obj = crud_mattermost.mattermost_channels.get_by_channel_name(
        db, team_name=team_name, channel_name=channel_name)
    if not channel_obj:
        raise HTTPException(
            status_code=422, detail="Mattermost channel not found")

    # TODO filter on time
    documents_obj = crud_mattermost.mattermost_documents.get_all_channel_documents(
        db, channels=[channel_obj.id])
    if not documents_obj:
        raise HTTPException(
            status_code=422, detail=f"Mattermost documents not found for channel: {channel_obj.id}")

    return documents_obj
