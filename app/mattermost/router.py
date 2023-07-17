"""mattermost router"""
from fastapi import APIRouter
from app.core.config import settings
import app.mattermost.mm_utils as mm
import json 
import pandas as pd
from fastapi import Depends
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.aimodels.bertopic.schemas.document import Document, DocumentCreate
from app.aimodels.bertopic import crud

"""mattermost section"""
router = APIRouter(
    prefix="", tags=["Mattermost"]
)


"""get a list of channels by mattermost user name"""
@router.get("/mattermost/user-info/", summary="Mattermost User Info", response_description="")
async def get_mm_user_info(user_name):
    """
    Get Mattermost user info.

    - **user_name**: Required.  Mattermost user name.
    """
    cdf = mm.get_all_user_channels(settings.mm_base_url, settings.mm_token, user_name)[['id', 'team_name', 'name']]
    # remove print statements after db integration completed
    print(cdf)
    print(cdf['id'].to_json())
    return cdf.to_json()


"""get a list of posts from a list of channel ids"""
@router.get("/mattermost/documents/", summary="Mattermost Documents", response_description="")
async def get_mm_channel_docs(channels, db: Session = Depends(get_db)):
    """
    Get Mattermost documents.

    - **channel_ids**: Required.  Mattermost channels to query for posts.
    """
    cjson = json.loads(channels)
    channel_ids = [cjson[key] for key in cjson]

    adf = pd.DataFrame()
    for channel_id in channel_ids:
        df = mm.get_channel_posts(settings.mm_base_url, settings.mm_token, channel_id)[['id', 'channel_id', 'user_id', 'create_at', 'message']]
        # remove print statements after db integration completed
        print(df)
        adf = pd.concat([adf, df], ignore_index=True)

    documents = [DocumentCreate(
            text=row['message'],
            mattermost_channel_id=row['channel_id'],
            mattermost_user_id=row['user_id'],
            mattermost_id=row['id'],
            original_created_time=row['create_at']
        ) for key, row in df.iterrows()]

    return crud.document.create_all_using_id(db, obj_in_list=documents)
