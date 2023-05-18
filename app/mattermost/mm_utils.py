"""mattermost utilities"""
import pandas as pd
import requests

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
    
    
def get_user_info(mm_base_url, mm_token, mm_user):
    
    # user info
    resp = requests.get(f'{mm_base_url}/api/v4/users/username/%s' % mm_user,\
                        headers={'Authorization': f'Bearer {mm_token}'})
    if resp.status_code < 400:
        buser = resp.json()
    #     print(json.dumps(buser, indent=2))
    else:
        print('request failed: %d' % resp.status_code)

    # team info
    url = f'{mm_base_url}/api/v4/users/%s/teams' % buser['id']
    return (buser, get_all_pages(url, mm_token))


def get_team_channels(mm_base_url, mm_token, user_id, team_id, team_name):

    url = f'{mm_base_url}/api/v4/users/%s/teams/%s/channels' % (user_id, team_id)
    df = get_all_pages(url, mm_token).assign(team_name=team_name)  
    return df[df.total_msg_count > 1]


def get_all_user_channels(mm_base_url, mm_token, mm_user):

    (user, bdf) = get_user_info(mm_base_url, mm_token, mm_user)
    cdf = pd.DataFrame()
    for index, team in bdf.iterrows():
        df = get_team_channels(mm_base_url, mm_token, user['id'], team['id'], team['name'])
        cdf = pd.concat([cdf, df], ignore_index=True)
    return cdf


def get_channel_posts(mm_base_url, mm_token, channel_id):
    
    url = f'{mm_base_url}/api/v4/channels/%s/posts' % channel_id
    return get_all_pages(url, mm_token, is_channel=True)


def get_all_user_channel_posts(mm_base_url, mm_token, channel_ids):
    
    df = pd.DataFrame()
    for cid in channel_ids:
        df = pd.concat([df, get_channel_posts(mm_base_url, mm_token, cid)])
    return df
    
def get_all_users(mm_base_url, mm_token):

    url = f'{mm_base_url}/api/v4/users'
    return get_all_pages(url, mm_token).set_index('username')


def get_all_public_teams(mm_base_url, mm_token):

    url = f'{mm_base_url}/api/v4/teams'
    return get_all_pages(url, mm_token).set_index('name')

