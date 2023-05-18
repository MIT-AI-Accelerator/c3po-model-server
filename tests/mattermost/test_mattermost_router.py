from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# returns 200 and nonempty body
def test_get_mattermost_user_info():
    response = client.get(
        "/mattermost/user-info/",
        headers={},
        params={"user_name":"nitmre-bot"},
    )

    assert response.status_code == 200
    assert response.json() != {}

# returns 200 and nonempty body
def test_get_mattermost_channel_posts():
    channel_ids = '{"0":"q3jroix947ymumrhn7mtz9uxmr","1":"z71qhruo9tyuujp7589sus6mew"}'
    response = client.get(
        "/mattermost/documents/",
        headers={},
        params={"channels":channel_ids},
    )

    assert response.status_code == 200
    assert response.json() != {}