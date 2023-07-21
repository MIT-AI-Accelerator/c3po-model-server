"""mattermost utilities"""
import pandas as pd
import requests


"""iterate through pages of an http request"""
def get_all_pages(url, mm_token, is_channel=False):
    
    per_page = 200
    page_num = 0
    rdf = pd.DataFrame()
    
    while True:
        resp = requests.get(url, headers={'Authorization': f'Bearer {mm_token}'},\
                            params={'page': page_num, 'per_page': per_page})
        print(resp.url)
        if resp.status_code < 400:                
            rdata = resp.json()
            if is_channel:
                rdata = rdata['posts']
                rdf = pd.concat([rdf, pd.DataFrame(rdata).transpose()], ignore_index=True)
            else:
                rdf = pd.concat([rdf, pd.DataFrame(rdata)], ignore_index=True)

            if len(rdata) < per_page:
                break
        else:
            print('request failed: %d' % resp.status_code)
            break
        page_num += 1
        
    return rdf
    
    
"""get a list of teams by user"""
def get_user_info(mm_base_url, mm_token, mm_user):
    
    # user info
    resp = requests.get(f'{mm_base_url}/api/v4/users/username/%s' % mm_user,\
                        headers={'Authorization': f'Bearer {mm_token}'})
    if resp.status_code < 400:
        user = resp.json()
    #     print(json.dumps(buser, indent=2))
    else:
        print('request failed: %d' % resp.status_code)

    # team info
    url = f'{mm_base_url}/api/v4/users/%s/teams' % user['id']
    tdf = get_all_pages(url, mm_token)
    if not tdf.empty:
        tdf.set_index('id', inplace=True)

    return (user, tdf)

"""get a list of teams by user"""
def get_user_name(mm_base_url, mm_token, mm_user):
    user_name = None

    # user info
    resp = requests.get(f'{mm_base_url}/api/v4/users/%s' % mm_user,\
                        headers={'Authorization': f'Bearer {mm_token}'})
    if resp.status_code < 400:
        user = resp.json()
        user_name = user['username']
    #     print(json.dumps(buser, indent=2))
    else:
        print('request failed: %d' % resp.status_code)

    return user_name

"""get a list of channels by team"""
# def get_team_channels(mm_base_url, mm_token, user_id, team_id, team_name):
def get_team_channels(mm_base_url, mm_token, user_id, team_id):

    url = f'{mm_base_url}/api/v4/users/%s/teams/%s/channels' % (user_id, team_id)
    # df = get_all_pages(url, mm_token).assign(team_name=team_name)
    df = get_all_pages(url, mm_token)
    return df[df.total_msg_count > 1]


"""get a list of channels by user"""
def get_all_user_channels(mm_base_url, mm_token, user_id, teams):

    cdf = pd.DataFrame()
    for tid in teams:
        df = get_team_channels(mm_base_url, mm_token, user_id, tid).assign(team_name=teams[tid])
        cdf = pd.concat([cdf, df])
    return cdf


"""get a list of posts for a single channel"""
def get_channel_posts(mm_base_url, mm_token, channel_id):
    
    url = f'{mm_base_url}/api/v4/channels/%s/posts' % channel_id
    return get_all_pages(url, mm_token, is_channel=True)


"""get a list of posts for a list of channels"""
def get_all_user_channel_posts(mm_base_url, mm_token, channel_ids):
    
    df = pd.DataFrame()
    for cid in channel_ids:
        df = pd.concat([df, get_channel_posts(mm_base_url, mm_token, cid)])
    return df
    
"""get a list of all users"""
def get_all_users(mm_base_url, mm_token):

    url = f'{mm_base_url}/api/v4/users'
    return get_all_pages(url, mm_token).set_index('username')


"""get a list of all public teams"""
def get_all_public_teams(mm_base_url, mm_token):

    url = f'{mm_base_url}/api/v4/teams'
    return get_all_pages(url, mm_token).set_index('name')
