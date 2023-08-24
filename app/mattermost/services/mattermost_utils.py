"""mattermost utilities"""
from datetime import datetime
import pandas as pd
import requests
from app.core.logging import logger

HTTP_REQUEST_TIMEOUT_S = 60
DEFAULT_HISTORY_DEPTH_DAYS = 45

def get_all_pages(url, mm_token, is_channel=False):
    """iterate through pages of an http request"""

    per_page = 200
    page_num = 0
    rdf = pd.DataFrame()

    while True:
        resp = requests.get(url, headers={'Authorization': f'Bearer {mm_token}'},
                            params={'page': page_num, 'per_page': per_page},
                            timeout=HTTP_REQUEST_TIMEOUT_S)
        if resp.status_code < 400:
            rdata = resp.json()
            if is_channel:
                rdata = rdata['posts']
                rdf = pd.concat(
                    [rdf, pd.DataFrame(rdata).transpose()], ignore_index=True)
            else:
                rdf = pd.concat([rdf, pd.DataFrame(rdata)], ignore_index=True)

            if len(rdata) < per_page:
                break
        else:
            logger.debug(f"{resp.url} request failed: {resp.status_code}")
            break
        page_num += 1

    return rdf


def get_user_info(mm_base_url, mm_token, mm_user):
    """get a list of teams by user"""

    user = None
    tdf = None

    # user info
    resp = requests.get(f'{mm_base_url}/api/v4/users/username/%s' % mm_user,
                        headers={'Authorization': f'Bearer {mm_token}'},
                        timeout=HTTP_REQUEST_TIMEOUT_S)
    if resp.status_code < 400:
        user = resp.json()
    else:
        logger.debug(f"{resp.url} request failed: {resp.status_code}")

    # team info
    if user:
        url = f'{mm_base_url}/api/v4/users/%s/teams' % user['id']
        tdf = get_all_pages(url, mm_token)
        if not tdf.empty:
            tdf.set_index('id', inplace=True)

    return (user, tdf)


def get_user_name(mm_base_url, mm_token, mm_user):
    """get a list of teams by user"""

    user_name = None

    # user info
    resp = requests.get(f'{mm_base_url}/api/v4/users/%s' % mm_user,
                        headers={'Authorization': f'Bearer {mm_token}'},
                        timeout=HTTP_REQUEST_TIMEOUT_S)
    if resp.status_code < 400:
        user = resp.json()
        user_name = user['username']
    else:
        logger.debug(f"{resp.url} request failed: {resp.status_code}")

    return user_name


def get_team_channels(mm_base_url, mm_token, user_id, team_id):
    """get a list of channels by team"""

    url = f'{mm_base_url}/api/v4/users/%s/teams/%s/channels' % (
        user_id, team_id)
    df = get_all_pages(url, mm_token)
    return df[df.total_msg_count > 1]


def get_all_user_channels(mm_base_url, mm_token, user_id, teams):
    """get a list of channels by user"""

    cdf = pd.DataFrame()
    for tid in teams:
        df = get_team_channels(mm_base_url, mm_token,
                               user_id, tid).assign(team_name=teams[tid])
        cdf = pd.concat([cdf, df])
    return cdf


def get_channel_info(mm_base_url, mm_token, channel_id):
    """get info for a single channel"""

    channel = None

    # channel info
    url = f'{mm_base_url}/api/v4/channels/%s' % channel_id
    resp = requests.get(url, headers={'Authorization': f'Bearer {mm_token}'},
                        timeout=HTTP_REQUEST_TIMEOUT_S)
    if resp.status_code < 400:
        channel = resp.json()
    else:
        logger.debug(f"{resp.url} request failed: {resp.status_code}")

    # team info
    if channel:
        url = f'{mm_base_url}/api/v4/teams/%s' % channel['team_id']
        resp = requests.get(url, headers={'Authorization': f'Bearer {mm_token}'},
                            timeout=HTTP_REQUEST_TIMEOUT_S)
        if resp.status_code < 400:
            team = resp.json()
            if team:
                channel['team_name'] = team['name']
        else:
            logger.debug(f"{resp.url} request failed: {resp.status_code}")

    return channel


def get_channel_posts(mm_base_url, mm_token, channel_id, history_depth=0):
    """get a list of posts for a single channel"""

    url = f'{mm_base_url}/api/v4/channels/%s/posts' % channel_id
    posts = get_all_pages(url, mm_token, is_channel=True)
    posts['datetime'] = [datetime.fromtimestamp(x / 1000) for x in posts['create_at']]

    if history_depth > 0:
        ctime = datetime.now()
        stime = ctime - pd.DateOffset(days=history_depth)
        posts = posts[(posts.datetime >= stime) & (posts.datetime <= ctime)]

    return posts


def get_all_user_channel_posts(mm_base_url, mm_token, channel_ids):
    """get a list of posts for a list of channels"""

    df = pd.DataFrame()
    for cid in channel_ids:
        df = pd.concat([df, get_channel_posts(mm_base_url, mm_token, cid)])
    return df


def get_all_users(mm_base_url, mm_token):
    """get a list of all users"""

    url = f'{mm_base_url}/api/v4/users'
    return get_all_pages(url, mm_token).set_index('username')


def get_all_public_teams(mm_base_url, mm_token):
    """get a list of all public teams"""

    url = f'{mm_base_url}/api/v4/teams'
    return get_all_pages(url, mm_token).set_index('name')
