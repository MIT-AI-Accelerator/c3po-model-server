"""mattermost utilities"""
from datetime import datetime
import pandas as pd
import requests

HTTP_REQUEST_TIMEOUT_S = 60
DEFAULT_HISTORY_DEPTH_DAYS = 45
MM_BOT_USERNAME = "nitmre-bot"

def get_all_pages(url, mm_token, is_channel=False, do_pagination=True):
    """iterate through pages of an http request"""

    per_page = 200
    page_num = 0
    rdf = pd.DataFrame()
    do_loop = True

    while do_loop:

        # workaround for mattermost pagination issue
        # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=44346553
        if not do_pagination:
            do_loop = False

        resp = requests.get(url, headers={'Authorization': f'Bearer {mm_token}'},
                            params={'page': page_num, 'per_page': per_page},
                            timeout=HTTP_REQUEST_TIMEOUT_S)
        if resp.status_code < 400:
            (rdf, rlen) = get_page_data(resp, rdf, per_page, is_channel)

            if rlen < per_page:
                break

        else:
            print(f"{resp.url} request failed: {resp.status_code}")
            break

        page_num += 1

    return rdf


def get_page_data(resp, rdf, per_page, is_channel):
    """get data for a single page of an http request"""

    rdata = resp.json()

    if is_channel:
        rdata = rdata['posts']
        rdf = pd.concat(
            [rdf, pd.DataFrame(rdata).transpose()], ignore_index=True)
    else:
        rdf = pd.concat([rdf, pd.DataFrame(rdata)], ignore_index=True)

    if len(rdata) > per_page:
        print(f"{resp.url} response length ({len(rdata)}) exceeds requested length ({per_page})")
    else:
        print(f"{resp.url} response length: {len(rdata)}")

    return (rdf, len(rdata))


def get_user_info(mm_base_url, mm_token, mm_user, get_teams = False):
    """get a list of teams by user"""

    user = None
    tdf = pd.DataFrame()

    # user info
    resp = requests.get(f'{mm_base_url}/api/v4/users/username/%s' % mm_user,
                        headers={'Authorization': f'Bearer {mm_token}'},
                        timeout=HTTP_REQUEST_TIMEOUT_S)
    if resp.status_code < 400:
        user = resp.json()
    else:
        print(f"{resp.url} request failed: {resp.status_code}")

    # team info
    if user and get_teams:
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
        print(f"{resp.url} request failed: {resp.status_code}")

    return user_name

def get_user_details(mm_base_url, mm_token, mm_user):
    """get user details by id"""

    udf = pd.DataFrame()

    # user info
    resp = requests.get(f'{mm_base_url}/api/v4/users/%s' % mm_user,
                        headers={'Authorization': f'Bearer {mm_token}'},
                        timeout=HTTP_REQUEST_TIMEOUT_S)
    if resp.status_code < 400:
        user = resp.json()
        del user['timezone'] # this dictionary complicates the dataframe init
        udf = pd.DataFrame([{'id': user['id'],
                            'user_name': user['username'],
                            'nickname': user['nickname'],
                            'first_name': user['first_name'],
                            'last_name': user['last_name'],
                            'position  ': user['position'],
                            'email': user['email']}])
    else:
        logger.debug(f"{resp.url} request failed: {resp.status_code}")

    return udf


def get_user_team_channels(mm_base_url, mm_token, user_id, team_id):
    """get a list of channels by team"""

    url = f'{mm_base_url}/api/v4/users/%s/teams/%s/channels' % (
        user_id, team_id)
    df = get_all_pages(url, mm_token, do_pagination=False)
    return df[df.total_msg_count > 0]


def get_team_channels(mm_base_url, mm_token, team_id):
    """get a list of channels by team"""

    url = f'{mm_base_url}/api/v4/teams/%s/channels' % team_id
    df = get_all_pages(url, mm_token, do_pagination=True)
    return df[df.total_msg_count > 0]


def get_all_user_team_channels(mm_base_url, mm_token, user_id, teams):
    """get a list of channels by user"""

    cdf = pd.DataFrame()
    for tid in teams:
        df = get_user_team_channels(mm_base_url, mm_token, user_id,
                                    tid).assign(team_name=teams[tid])
        cdf = pd.concat([cdf, df])
    return cdf


def get_all_team_channels(mm_base_url, mm_token, teams):
    """get a list of channels by user"""

    cdf = pd.DataFrame()
    for tid in teams:
        df = get_team_channels(mm_base_url, mm_token,
                               tid).assign(team_name=teams[tid])
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
        print(f"{resp.url} request failed: {resp.status_code}")

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
            print(f"{resp.url} request failed: {resp.status_code}")

    return channel


def get_channel_posts(mm_base_url, mm_token, channel_id, history_depth=0, usernames_to_filter=set([MM_BOT_USERNAME])):
    """get a list of posts for a single channel"""

    url = f'{mm_base_url}/api/v4/channels/%s/posts' % channel_id
    posts = get_all_pages(url, mm_token, is_channel=True)
    if not posts.empty:
        posts['datetime'] = [datetime.fromtimestamp(x / 1000) for x in posts['create_at']]

        if history_depth > 0:
            ctime = datetime.now()
            stime = ctime - pd.DateOffset(days=history_depth)
            posts = posts[(posts.datetime >= stime) & (posts.datetime <= ctime)]

        user_ids_to_filter = set()
        for mm_user in usernames_to_filter:
            (user, tdf) = get_user_info(mm_base_url, mm_token, mm_user)
            if user:
                user_ids_to_filter.add(user['id'])
            else:
                print(f"skipping user filter for channel {channel_id}, unable to find mattermost id for {mm_user}")

        user_ids_to_filter = user_ids_to_filter.intersection(posts['user_id'].unique())
        posts = posts[~posts['user_id'].isin(user_ids_to_filter)]

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
