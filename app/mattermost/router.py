"""mattermost router"""
from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
import pandas as pd
from sqlalchemy.orm import Session
from ppg.schemas.bertopic.document import Document, DocumentCreate
from ppg.schemas.mattermost.mattermost_documents import MattermostDocument, MattermostDocumentCreate
from ppg.schemas.mattermost.mattermost_users import MattermostUser
import ppg.services.mattermost_utils as mattermost_utils
from app.core.config import settings
from app.core.errors import HTTPValidationError
from app.core.logging import logger
from app.dependencies import get_db
from app.aimodels.bertopic import crud as crud_document
from app.mattermost.crud import crud_mattermost

"""mattermost section"""
router = APIRouter(
    prefix="", tags=["Mattermost - Experimental"]
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

    user_obj = crud_mattermost.populate_mm_user_team_info(
        db, user_name=request.user_name, get_teams=True)
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
    history_depth: int = mattermost_utils.DEFAULT_HISTORY_DEPTH_DAYS
    filter_bot_posts = True


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
    - **history_depth**: Optional.  Number of days (prior to request) to upload posts for.
    - **filter_bot_posts**: Optional.  Eliminate bot posts from upload.
    """

    if not request.channel_ids:
        raise HTTPException(
            status_code=422, detail="No Mattermost channels requested")

    usernames_to_filter = set()
    if request.filter_bot_posts:
        usernames_to_filter.add(mattermost_utils.MM_BOT_USERNAME)

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
            settings.mm_base_url,
            settings.mm_token,
            channel_id,
            request.history_depth,
            usernames_to_filter).assign(channel=channel_obj.id)
        adf = pd.concat([adf, df], ignore_index=True)
    channel_uuids = adf['channel'].unique()

    # handle empty channels
    # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=44143308
    if not adf.empty:
        user_ids = adf['user_id'].unique()
        for uid in user_ids:
            user_obj = crud_mattermost.mattermost_users.get_by_user_id(
                db, user_id=uid)
            if user_obj is None:
                user_name = mattermost_utils.get_user_name(
                    settings.mm_base_url, settings.mm_token, uid)
                user_obj = crud_mattermost.populate_mm_user_team_info(
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
            type=row['type'],
            hashtags=row['hashtags'],
            has_reactions=str(row['has_reactions']).lower() == 'true',
            props=row['props'],
            doc_metadata=row['metadata'],
            channel=row['channel'],
            user=row['user'],
            document=document_obj.id)]
    crud_mattermost.mattermost_documents.create_all_using_id(db, obj_in_list=mattermost_documents)

    return crud_mattermost.mattermost_documents.get_all_channel_documents(
        db, channels=channel_uuids, history_depth=request.history_depth)


@router.get(
    "/mattermost/documents/get",
    response_model=Union[list[MattermostDocument], HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve Mattermost documents",
    response_description="Retrieved Mattermost documents")
async def get_mm_channel_docs(team_name: str, channel_name: str,
                              history_depth: int = mattermost_utils.DEFAULT_HISTORY_DEPTH_DAYS,
                              db: Session = Depends(get_db)) -> (
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

    documents_obj = crud_mattermost.mattermost_documents.get_all_channel_documents(
        db, channels=[channel_obj.id], history_depth=history_depth)
    if not documents_obj:
        raise HTTPException(
            status_code=422, detail=f"Mattermost documents not found for channel: {channel_obj.id}")

    return documents_obj

class ConversationThreadRequest(BaseModel):
    mattermost_document_ids: list[UUID4] = []

@router.post(
    "/mattermost/conversation_threads",
    response_model=Union[list[Document], HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve Mattermost conversation documents",
    response_description="Retrieved Mattermost conversation documents")
async def convert_conversation_threads(request: ConversationThreadRequest,
                              db: Session = Depends(get_db)) -> (
    Union[list[Document], HTTPValidationError]
):
    """
    Retrieve Mattermost conversation documents

    - **mattermost_document_uuids**: Required.  List of Mattermost document UUIDs.
    """

    # get joined document and mattermost info
    document_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids=request.mattermost_document_ids)
    if document_df.empty:
        raise HTTPException(status_code=422, detail="Mattermost documents not found")

    # convert message utterances to conversation threads
    conversation_df = crud_mattermost.convert_conversation_threads(df=document_df)

    # create new document objects
    document_objs = [crud_document.document.create(db, obj_in=DocumentCreate(text=row['message'], original_created_time=row['create_at'])) for key, row in conversation_df.iterrows()]
    if not document_objs:
        raise HTTPException(status_code=422, detail="Unable to create conversation threads")

    return document_objs
