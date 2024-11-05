"""mattermost router"""
from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
from tqdm import tqdm
import pandas as pd
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.errors import HTTPValidationError
from app.dependencies import get_db
from app.aimodels.bertopic import crud as crud_document
from app.mattermost.crud import crud_mattermost
from app.ppg_common.schemas.bertopic.document import DocumentUpdate
from app.ppg_common.schemas.mattermost.mattermost_documents import MattermostDocument, MattermostDocumentUpdate, InfoTypeEnum, ThreadTypeEnum
from app.ppg_common.schemas.mattermost.mattermost_users import MattermostUser
import app.ppg_common.services.mattermost_utils as mattermost_utils

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
    filter_bot_posts = False
    filter_system_posts = True
    filter_post_content: list[str] = []

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
    - **filter_system_posts**: Optional.  Eliminate system posts from upload.
    - **filter_post_content**: Optional.  A list of bot properties to include, e.g. "chat".
    """

    if not request.channel_ids:
        raise HTTPException(
            status_code=422, detail="No Mattermost channels requested")

    filters = [f.value for f in InfoTypeEnum]
    if any(f not in filters for f in request.filter_post_content):
        raise HTTPException(
            status_code=422, detail="Invalid post content filter: %s. Available filters: %s" %
            ([f for f in request.filter_post_content], filters))

    usernames_to_filter = set()
    if request.filter_bot_posts:
        usernames_to_filter.add(mattermost_utils.MM_BOT_USERNAME)

    adf = pd.DataFrame()
    for channel_id in tqdm(request.channel_ids):
        channel_obj = crud_mattermost.get_or_create_mm_channel_object(db, channel_id=channel_id)
        df = mattermost_utils.get_channel_posts(
            settings.mm_base_url,
            settings.mm_token,
            channel_id,
            history_depth=request.history_depth,
            filter_system_types=request.filter_system_posts,
            usernames_to_filter=usernames_to_filter).assign(channel=channel_obj.id)
        adf = pd.concat([adf, df], ignore_index=True)
    channel_uuids = adf['channel'].unique()

    # handle empty channels
    # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=44143308
    if not adf.empty:
        user_ids = adf['user_id'].unique()
        for uid in user_ids:
            user_obj = crud_mattermost.get_or_create_mm_user_object(db, user_id=uid)
            adf.loc[adf['user_id'] == uid, 'user'] = user_obj.id

        channel_document_objs = crud_mattermost.mattermost_documents.get_all_channel_documents(
            db, channels=channel_uuids)
        existing_ids = [obj.message_id for obj in channel_document_objs]
        adf = adf[~adf.id.isin(existing_ids)].drop_duplicates(subset='id')

    adf.rename(columns={'id': 'message_id'}, inplace=True)
    crud_mattermost.mattermost_documents.create_all_using_df(db, ddf=adf, thread_type=ThreadTypeEnum.MESSAGE)

    return crud_mattermost.mattermost_documents.get_all_channel_documents(db,
                                                                          channels=channel_uuids,
                                                                          history_depth=request.history_depth,
                                                                          content_filter_list=request.filter_post_content)


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

class ConversationThreadResponse(BaseModel):
    threads: list[MattermostDocument] = []
    threads_speaker: list[MattermostDocument] = []
    threads_speaker_persona: list[MattermostDocument] = []

@router.post(
    "/mattermost/conversation_threads",
    response_model=Union[ConversationThreadResponse, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve Mattermost conversation documents",
    response_description="Retrieved Mattermost conversation documents")
async def convert_conversation_threads(request: ConversationThreadRequest,
                              db: Session = Depends(get_db)) -> (
    Union[ConversationThreadResponse, HTTPValidationError]
):
    """
    Retrieve Mattermost conversation documents

    - **mattermost_document_uuids**: Required.  List of Mattermost document UUIDs.
    """

    # get joined document and mattermost info
    document_df = crud_mattermost.mattermost_documents.get_mm_document_dataframe(db, mm_document_uuids=request.mattermost_document_ids)
    if document_df.empty:
        raise HTTPException(status_code=422, detail="Mattermost documents not found")

    # only thread user chat messages. other types, such as ACARS, NOTAMS should remain unthreaded
    chat_df = document_df[document_df['info_type'] == InfoTypeEnum.CHAT]
    other_df = document_df[document_df['info_type'] != InfoTypeEnum.CHAT]

    # convert message utterances to conversation threads
    conversation_df = crud_mattermost.convert_conversation_threads(df=chat_df)
    conversation_df.rename(columns={'user_uuid': 'user','channel_uuid': 'channel'}, inplace=True)

    other_mm_doc_objs = [crud_mattermost.mattermost_documents.get_by_message_id(db, message_id=row['message_id'])
                         for key, row in other_df.iterrows()]
    if not other_df.empty and (len(other_mm_doc_objs) != len(other_df)):
        raise HTTPException(status_code=422, detail="Unable to find non chat documents")

    thread_document_objs = ConversationThreadResponse()
    thread_document_objs.threads = create_conversation_objects(db=db,
                                                               thread_type=ThreadTypeEnum.THREAD,
                                                               conversation_df=conversation_df) + other_mm_doc_objs

    thread_document_objs.threads_speaker = create_conversation_objects(db=db,
                                                               thread_type=ThreadTypeEnum.THREAD_USER,
                                                               conversation_df=conversation_df) + other_mm_doc_objs
    thread_document_objs.threads_speaker_persona = create_conversation_objects(db=db,
                                                               thread_type=ThreadTypeEnum.THREAD_USER_PERSONA,
                                                               conversation_df=conversation_df) + other_mm_doc_objs

    return thread_document_objs

def create_conversation_objects(db: Session, thread_type: ThreadTypeEnum, conversation_df: pd.DataFrame) -> list[MattermostDocument]:

    thread_document_objs = []
    thread_df = pd.DataFrame()

    for _, row in conversation_df.iterrows():

        thread_str = row['thread']
        if thread_type == ThreadTypeEnum.THREAD_USER:
            thread_str = row['thread_speaker']
        if thread_type == ThreadTypeEnum.THREAD_USER_PERSONA:
            thread_str = row['thread_speaker_persona']

        mm_document_obj = crud_mattermost.mattermost_documents.get_by_message_id(db, message_id=row['message_id'], thread_type=thread_type)

        # update existing thread
        if mm_document_obj:
            document_obj = crud_document.document.get(db, id=row['document_id'])
            crud_document.document.update(db,
                                 db_obj=document_obj,
                                 obj_in=DocumentUpdate(text=thread_str,
                                                       original_created_time=document_obj.original_created_time))
            updated_mm_doc_obj = crud_mattermost.mattermost_documents.update(db,
                                                                             db_obj=mm_document_obj,
                                                                             obj_in=MattermostDocumentUpdate(
                                                                                message_id=mm_document_obj.message_id,
                                                                                root_message_id=mm_document_obj.root_message_id,
                                                                                type=mm_document_obj.type,
                                                                                hashtags=row['hashtags'],
                                                                                has_reactions=str(row['has_reactions']).lower() == 'true',
                                                                                props=row['props'],
                                                                                doc_metadata=row['metadata'],
                                                                                channel=mm_document_obj.channel,
                                                                                user=mm_document_obj.user,
                                                                                document=mm_document_obj.document,
                                                                                thread_type=thread_type,
                                                                                info_type=mm_document_obj.info_type))
            thread_document_objs = thread_document_objs + [updated_mm_doc_obj]

        else:
            row['message'] = thread_str
            thread_df = pd.concat([thread_df, pd.DataFrame([row])])

    # create new thread objects
    if not thread_df.empty:
        new_mm_doc_objs = crud_mattermost.mattermost_documents.create_all_using_df(db, ddf=thread_df, thread_type=thread_type)
        thread_document_objs = thread_document_objs + new_mm_doc_objs

    if len(thread_document_objs) != len(conversation_df):
        raise HTTPException(status_code=422, detail="Unable to create conversation threads")

    return thread_document_objs

class SubstringUploadRequest(BaseModel):
    team_id: str
    search_terms: list[str]


@router.post(
    "/mattermost/search/upload",
    response_model=dict,
    responses={'422': {'model': HTTPValidationError}},
    summary="Upload Mattermost documents containing substring",
    response_description="Uploaded Mattermost documents containing substring")
async def upload_mm_docs_by_substring(request: SubstringUploadRequest, db: Session = Depends(get_db)) -> dict:
    """
    Retrieve mattermost posts by substring

    - **team_id**: Required.  Team ID for post query.
    - **search_terms**: Required.  List of substrings for post query.
    """
    existing_doc_uuids = []
    new_message_ids = []
    ddf = pd.DataFrame()
    for search_str in tqdm(request.search_terms):
        ddf = pd.concat([ddf,
                         mattermost_utils.get_all_team_posts_by_substring(settings.mm_base_url, settings.mm_token, request.team_id, search_str)],
                         ignore_index=True)
    ddf.drop_duplicates(subset=['id'], inplace=True)
    for key, row in ddf.iterrows():
        mm_doc = crud_mattermost.mattermost_documents.get_by_message_id(db, message_id=row.id)
        if mm_doc:
            existing_doc_uuids.append(mm_doc.document)
        else:
            new_message_ids.append(row.id)
    ddf = ddf[ddf.id.isin(new_message_ids)]

    new_mattermost_docs = crud_mattermost.populate_mm_document_info(db, document_df=ddf)
    new_doc_uuids = [mm_doc.document for mm_doc in new_mattermost_docs]

    return crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids=(existing_doc_uuids + new_doc_uuids)).transpose().to_dict()


@router.get(
    "/mattermost/search/get",
    response_model=dict,
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve Mattermost documents containing substring",
    response_description="Retrieved Mattermost documents containing substring")
async def get_mm_docs_by_substring(search_terms: str, db: Session = Depends(get_db)) -> dict:
    """
    Retrieve mattermost posts by substring

    - **search_terms**: Required.  Comma-separated list of case-insensitive substrings for post query.
    """
    search_terms = search_terms.split(',')

    ddf = pd.DataFrame()
    for search_str in tqdm(search_terms):
        ddf = pd.concat([ddf,
                         crud_mattermost.mattermost_documents.get_by_substring(db, search_str=search_str)],
                         ignore_index=True)
    ddf.drop_duplicates(subset=['link'], inplace=True)

    return ddf.transpose().to_dict()
