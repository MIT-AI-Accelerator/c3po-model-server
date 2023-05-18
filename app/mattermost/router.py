from fastapi import APIRouter
from app.core.config import settings
import app.mattermost.mm_utils as mm
import json 
import pandas as pd

router = APIRouter(
    prefix="", tags=["Mattermost"]
)

@router.get("/user-info/", summary="Mattermost User Info", response_description="")
async def get_mm_user_info(user_name):
    """
    Get Mattermost user info.

    - **user_name**: Required.  Mattermost user name.
    """
    cdf = mm.get_all_user_channels(settings.mm_base_url, settings.mm_token, user_name)[['id', 'team_name', 'name']]
    print(cdf)
    print(cdf['id'].to_json())
    return cdf.to_json()


@router.get("/documents/", summary="Mattermost Documents", response_description="")
async def get_mm_channel_docs(channels):
    """
    Get Mattermost documents.

    - **channel_ids**: Required.  Mattermost channels to query for posts.
    """
    cjson = json.loads(channels)
    # print(json.dumps(cjson, indent=2))
    channel_ids = [cjson[key] for key in cjson]
    # print(channel_ids)

    adf = pd.DataFrame()
    # channel_ids = ['z71qhruo9tyuujp7589sus6mew', 'pi7g6boebiyd8nt81hrbudsq3r', 'aw86jqw4jinbtqbxn8xq3qhduo', '5qzzh6hfifdd8x39handaq3qhc', 'xpgjjdkyq78m5fpgcgd14sxf5w']
    for channel_id in channel_ids:
        df = mm.get_channel_posts(settings.mm_base_url, settings.mm_token, channel_id)[['channel_id', 'create_at', 'message']]
        print(df)
        adf = pd.concat([adf, df], ignore_index=True)
    return adf.to_json()

