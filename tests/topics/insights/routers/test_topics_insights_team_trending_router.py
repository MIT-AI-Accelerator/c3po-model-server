from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# returns 200 and nonempty body
def test_get_basic_chat_team_vizualization():
    response = client.post(
        "/topics/insights/teamtrending/visualization/",
        headers={},
        json={"team":"nitmre"},
    )

    assert response.status_code == 200
    assert response.json() != {}
