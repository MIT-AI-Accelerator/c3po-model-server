from pytest import mark
from fastapi.testclient import TestClient

@mark.parametrize('label, answer', [
    ('review', 'medium'),
    ('action', 'high'),
    ('recycle', 'low')
])
def test_get_single_line_chat_stress(client: TestClient,
                                     label: str,
                                     answer: str):
    """Tests the mapping of labels to answers performed in the router.
    Sends the label in place of the chat message text that would get sent
    to the model for classification."""

    response = client.post(
        "/sentiments/insights/getchatstress",
        headers={},
        json={'text': label},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": answer}
