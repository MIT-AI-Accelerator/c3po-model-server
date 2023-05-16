
from app.core.config import settings
from app.dependencies import httpx_client

# see mm api spec here: https://api.mattermost.com/#tag/posts/operation/GetPostsForChannel
# see httpx docs here: https://www.python-httpx.org/
async def get_posts_by_channel(channel_id: str):
    url = f'{settings.mm_base_url}/api/v4/channels/{channel_id}/posts?per_page'

    headers = {'Authorization': f'Bearer {settings.mm_token}'}
    params = {'per_page': '1000'}

    req = httpx_client.build_request('GET', url, headers=headers, params=params)
    return await httpx_client.send(req, stream=True)
