import uuid
import pandas as pd
from unittest.mock import MagicMock, patch
from app.core.config import settings, environment_settings
from app.ppg_common.services import mattermost_utils

def test_mattermost_bot():
# test mattermost api calls for user, team, and channel info

    if (environment_settings.environment == 'test') or (environment_settings.environment == 'integration'):
        return

    # test get user, teams
    mm_name = 'nitmre-bot'
    user, teams = mattermost_utils.get_user_info(
        settings.mm_base_url, settings.mm_token, mm_name, get_teams = True)

    assert user['username'] == mm_name
    assert not teams.empty
    assert mattermost_utils.get_user_name(
        settings.mm_base_url, settings.mm_token, user['id']) == mm_name

    # test get channels
    teams = teams[teams['name'] == 'usaf-618aoc-mod']
    channels = mattermost_utils.get_user_team_channels(
        settings.mm_base_url, settings.mm_token, user['id'], teams.index[0])
    assert not channels.empty

    public_channels = channels[(channels['type'] == 'O') & (channels['total_msg_count'] > 1)]
    smallest_channel = public_channels.loc[public_channels['total_msg_count'].idxmin()]
    assert smallest_channel['total_msg_count'] == 2

    channel_info = mattermost_utils.get_channel_info(
        settings.mm_base_url, settings.mm_token, smallest_channel['id'])
    assert channel_info['name'] == smallest_channel['name']

    # test get documents
    documents = mattermost_utils.get_channel_posts(
        settings.mm_base_url, settings.mm_token, smallest_channel['id'], filter_system_types=False, usernames_to_filter=set())
    # 20241223 mattermost api bug - channel total_msg_count does not equal number of posts returned
    # assert len(documents) == smallest_channel['total_msg_count']
    assert len(documents) > 0


def test_get_user_info(mocker):
# pipeline safe test for get_user_info

    user = str(uuid.uuid4())
    mock_data = (dict({'id': user, 'username': user}), pd.DataFrame())

    mocker.patch('app.ppg_common.services.mattermost_utils.get_user_info', return_value=mock_data)

    user_info = mattermost_utils.get_user_info("127.0.0.1", "a_token", user)

    assert user_info == mock_data


def test_get_all_user_channels(mocker):
# pipeline safe test for get_all_user_team_channels

    user = str(uuid.uuid4())
    team = str(uuid.uuid4())
    mock_data = pd.DataFrame()

    mocker.patch('app.ppg_common.services.mattermost_utils.get_user_team_channels', return_value=mock_data)
    mocker.patch('app.ppg_common.services.mattermost_utils.get_all_user_team_channels', return_value=mock_data)

    channel_info = mattermost_utils.get_all_user_team_channels("127.0.0.1", "a_token", user, [team])

    assert channel_info.equals(mock_data)


def test_http_request_timeout_constant():
    """Test that HTTP_REQUEST_TIMEOUT_S constant is defined"""
    assert hasattr(mattermost_utils, 'HTTP_REQUEST_TIMEOUT_S')
    assert mattermost_utils.HTTP_REQUEST_TIMEOUT_S == 60


def test_default_history_depth_constant():
    """Test that DEFAULT_HISTORY_DEPTH_DAYS constant is defined"""
    assert hasattr(mattermost_utils, 'DEFAULT_HISTORY_DEPTH_DAYS')
    assert mattermost_utils.DEFAULT_HISTORY_DEPTH_DAYS == 45


def test_mm_bot_username_constant():
    """Test that MM_BOT_USERNAME constant is defined"""
    assert hasattr(mattermost_utils, 'MM_BOT_USERNAME')
    assert mattermost_utils.MM_BOT_USERNAME == "nitmre-bot"


def test_get_page_data_not_channel():
    """Test get_page_data with non-channel data"""
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{'id': '1', 'name': 'test'}]
    mock_resp.url = "http://test.com"
    
    rdf = pd.DataFrame()
    per_page = 200
    
    result_df, result_len = mattermost_utils.get_page_data(mock_resp, rdf, per_page, is_channel=False)
    
    assert len(result_df) == 1
    assert result_len == 1


def test_get_page_data_with_channel():
    """Test get_page_data with channel data"""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        'posts': {
            'post1': {'id': 'post1', 'message': 'test message'},
            'post2': {'id': 'post2', 'message': 'another message'}
        }
    }
    mock_resp.url = "http://test.com"
    
    rdf = pd.DataFrame()
    per_page = 200
    
    result_df, result_len = mattermost_utils.get_page_data(mock_resp, rdf, per_page, is_channel=True)
    
    assert len(result_df) == 2
    assert result_len == 2


def test_get_page_data_exceeds_per_page(caplog):
    """Test get_page_data logs warning when response exceeds per_page"""
    mock_resp = MagicMock()
    # Create data that exceeds per_page
    large_data = [{'id': str(i)} for i in range(250)]
    mock_resp.json.return_value = large_data
    mock_resp.url = "http://test.com"
    
    rdf = pd.DataFrame()
    per_page = 200
    
    result_df, result_len = mattermost_utils.get_page_data(mock_resp, rdf, per_page, is_channel=False)
    
    assert result_len == 250
    assert "exceeds requested length" in caplog.text


def test_get_all_pages_single_page(mocker):
    """Test get_all_pages with single page of results"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{'id': '1', 'name': 'test'}]
    mock_resp.url = "http://test.com"
    
    mocker.patch('app.ppg_common.services.mattermost_utils.requests.get', return_value=mock_resp)
    
    result = mattermost_utils.get_all_pages("http://test.com", "token", is_channel=False)
    
    assert len(result) == 1


def test_get_all_pages_with_pagination_disabled(mocker):
    """Test get_all_pages with do_pagination=False"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{'id': '1'}] * 200
    mock_resp.url = "http://test.com"
    
    mocker.patch('app.ppg_common.services.mattermost_utils.requests.get', return_value=mock_resp)
    
    result = mattermost_utils.get_all_pages("http://test.com", "token", is_channel=False, do_pagination=False)
    
    # Should only fetch one page
    assert len(result) == 200


def test_get_all_pages_http_error(mocker, caplog):
    """Test get_all_pages handles HTTPError"""
    import requests
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")
    mock_resp.headers = {}
    mock_resp.text = "Error"
    
    mocker.patch('app.ppg_common.services.mattermost_utils.requests.get', return_value=mock_resp)
    
    result = mattermost_utils.get_all_pages("http://test.com", "token", is_channel=False)
    
    assert len(result) == 0
    assert "request failed" in caplog.text


def test_get_all_pages_timeout(mocker, caplog):
    """Test get_all_pages handles ReadTimeout"""
    import requests
    mock_resp = MagicMock()
    mock_resp.status_code = 408
    mock_resp.raise_for_status.side_effect = requests.exceptions.ReadTimeout("Timeout")
    mock_resp.headers = {}
    mock_resp.text = "Timeout"
    
    mocker.patch('app.ppg_common.services.mattermost_utils.requests.get', return_value=mock_resp)
    
    result = mattermost_utils.get_all_pages("http://test.com", "token", is_channel=False)
    
    assert len(result) == 0
    assert "timed out" in caplog.text


def test_get_user_info_success(mocker):
    """Test get_user_info with successful response"""
    mock_resp_user = MagicMock()
    mock_resp_user.status_code = 200
    mock_resp_user.json.return_value = {'id': 'user123', 'username': 'testuser'}
    
    mocker.patch('app.ppg_common.services.mattermost_utils.requests.get', return_value=mock_resp_user)
    
    user, teams = mattermost_utils.get_user_info("http://test.com", "token", "testuser", get_teams=False)
    
    assert user['id'] == 'user123'
    assert user['username'] == 'testuser'
    assert teams.empty


def test_get_user_info_failure(mocker, caplog):
    """Test get_user_info with failed response"""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.url = "http://test.com"
    
    mocker.patch('app.ppg_common.services.mattermost_utils.requests.get', return_value=mock_resp)
    
    user, teams = mattermost_utils.get_user_info("http://test.com", "token", "baduser", get_teams=False)
    
    assert user is None
    assert "request failed" in caplog.text
