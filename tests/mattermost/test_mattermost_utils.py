import uuid
import pandas as pd
from unittest.mock import patch, Mock, create_autospec
from app.core.config import settings, environment_settings
from app.mattermost.services import mattermost_utils

def test_mattermost_bot():
# test mattermost api calls for user, team, and channel info

    if environment_settings.environment == 'test':
        return

    # test get user, teams
    mm_name = 'nitmre-bot'
    user, teams = mattermost_utils.get_user_info(
        settings.mm_base_url, settings.mm_token, mm_name)

    assert user['username'] == mm_name
    assert not teams.empty
    assert mattermost_utils.get_user_name(
        settings.mm_base_url, settings.mm_token, user['id']) == mm_name

    # test get channels
    channels = mattermost_utils.get_all_user_channels(
        settings.mm_base_url, settings.mm_token, user['id'], teams['name'].to_dict())

    assert not channels.empty
    assert mattermost_utils.get_channel_info(
        settings.mm_base_url, settings.mm_token, channels['id'].iloc[0])['name'] == channels['name'].iloc[0]

    # test get documents
    documents = mattermost_utils.get_channel_posts(
        settings.mm_base_url, settings.mm_token, channels['id'].iloc[0])

    assert not documents.empty


def test_get_user_info(mocker):
# pipeline safe test for get_user_info

    user = str(uuid.uuid4())
    mock_data = (dict({'id': user, 'username': user}), pd.DataFrame())

    mocker.patch('app.mattermost.services.mattermost_utils.get_user_info', return_value=mock_data)

    user_info = mattermost_utils.get_user_info("127.0.0.1", "a_token", user)

    assert user_info == mock_data


def test_get_all_user_channels(mocker):
# pipeline safe test for get_all_user_channels

    user = str(uuid.uuid4())
    team = str(uuid.uuid4())
    mock_data = pd.DataFrame()

    mocker.patch('app.mattermost.services.mattermost_utils.get_team_channels', return_value=mock_data)
    mocker.patch('app.mattermost.services.mattermost_utils.get_all_user_channels', return_value=mock_data)

    channel_info = mattermost_utils.get_all_user_channels("127.0.0.1", "a_token", user, [team])

    assert channel_info.equals(mock_data)
