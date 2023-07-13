from fastapi.testclient import TestClient

# Given: A line of chat--"Hello there!" and the above mocked model that classifies it as "recycle"
# When: This line is sent to the endpoint /predict
# Then: we expect to receive a 200 and the appropriately formatted response in the body
def test_get_single_line_chat_stress(client: TestClient):

    response = client.post(
        "/sentiments/insights/getchatstress",
        headers={},
        json={"text": "Hello there!"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "low"}
