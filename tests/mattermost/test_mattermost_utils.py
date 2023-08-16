from app.core.config import settings, environment_settings
from app.mattermost.services import mattermost_utils


# test mattermost api calls for user, team, and channel info
def test_mattermost_bot():

    if environment_settings.environment == 'test':
        return

    # test get user, teams
    mm_name = 'nitmre-bot'
    user, teams = mattermost_utils.get_user_info(
        settings.mm_base_url, settings.mm_token, mm_name)

    assert user['username'] == mm_name
    assert not teams.empty

    # test get channels
    channels = mattermost_utils.get_all_user_channels(
        settings.mm_base_url, settings.mm_token, user['id'], teams['name'].to_dict())

    assert not channels.empty

    # test get documents
    documents = mattermost_utils.get_channel_posts(
        settings.mm_base_url, settings.mm_token, channels['id'].iloc[0])

    assert not documents.empty
