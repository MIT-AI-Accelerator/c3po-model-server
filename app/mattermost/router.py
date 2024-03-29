"""mattermost router"""
from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
import pandas as pd
from sqlalchemy.orm import Session
from ppg.schemas.bertopic.document import DocumentUpdate
from ppg.schemas.mattermost.mattermost_documents import MattermostDocument, MattermostDocumentUpdate
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

    adf.rename(columns={'id': 'message_id'}, inplace=True)
    crud_mattermost.mattermost_documents.create_all_using_df(db, ddf=adf, is_thread=False)

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
    response_model=Union[list[MattermostDocument], HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve Mattermost conversation documents",
    response_description="Retrieved Mattermost conversation documents")
async def convert_conversation_threads(request: ConversationThreadRequest,
                              db: Session = Depends(get_db)) -> (
    Union[list[MattermostDocument], HTTPValidationError]
):
    """
    Retrieve Mattermost conversation documents

    - **mattermost_document_uuids**: Required.  List of Mattermost document UUIDs.
    """

    # get joined document and mattermost info
    document_df = crud_mattermost.mattermost_documents.get_mm_document_dataframe(db, mm_document_uuids=request.mattermost_document_ids)
    if document_df.empty:
        raise HTTPException(status_code=422, detail="Mattermost documents not found")

    # convert message utterances to conversation threads
    conversation_df = crud_mattermost.convert_conversation_threads(df=document_df)
    conversation_df.rename(columns={'user_uuid': 'user','channel_uuid': 'channel'}, inplace=True)

    document_objs = []
    new_threads_df = pd.DataFrame()
    for _, row in conversation_df.iterrows():
        mm_document_obj = crud_mattermost.mattermost_documents.get_by_message_id(db, message_id=row['message_id'], is_thread=True)

        # update existing thread
        if mm_document_obj:
            document_obj = crud_document.document.get(db, id=row['document_id'])
            crud_document.document.update(db,
                                 db_obj=document_obj,
                                 obj_in=DocumentUpdate(text=row['message'],
                                                       original_created_time=document_obj.original_created_time))
            updated_mm_doc_obj = crud_mattermost.mattermost_documents.update(db,
                                                                             db_obj=mm_document_obj,
                                                                             obj_in=MattermostDocumentUpdate(message_id=mm_document_obj.message_id,
                                                                             root_message_id=mm_document_obj.root_message_id,
                                                                             type=mm_document_obj.type,hashtags=row['hashtags'],
                                                                             has_reactions=str(row['has_reactions']).lower() == 'true',
                                                                             props=row['props'],
                                                                             doc_metadata=row['metadata'],
                                                                             channel=mm_document_obj.channel,
                                                                             user=mm_document_obj.user,
                                                                             document=mm_document_obj.document,
                                                                             is_thread=True))
            document_objs = document_objs + [updated_mm_doc_obj]

        else:
            new_threads_df = pd.concat([new_threads_df, pd.DataFrame([row])])

    # create new thread objects
    if not new_threads_df.empty:
        new_mm_doc_objs = crud_mattermost.mattermost_documents.create_all_using_df(db, ddf=new_threads_df, is_thread=True)
        document_objs = document_objs + new_mm_doc_objs

    if len(document_objs) != len(conversation_df):
        raise HTTPException(status_code=422, detail="Unable to create conversation threads")

    return document_objs
