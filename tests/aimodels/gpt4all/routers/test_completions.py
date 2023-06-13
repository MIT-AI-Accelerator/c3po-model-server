from fastapi.testclient import TestClient
from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInferenceInputs
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllModelFilenameEnum, Gpt4AllPretrainedModel

from app.main import app
from app.aimodels.bertopic.routers.documents import get_db
from tests.test_files.db.db_test_session import SessionLocal

from tests.test_files.mocks.mock_completion_inference import gen_mock_completion_inference

# ************Mocks*******************
def mock_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = mock_db
# *************************************


def test_gpt_completion_post_valid_input(client: TestClient, mocker):

    MockCompletionInferenceObject, MockCompletionInference = gen_mock_completion_inference(
        "This is a test response")

    # patch the CompletionInference class with the MockCompletionInference
    mocker.patch("app.aimodels.gpt4all.routers.completions.CompletionInference",
                 new=MockCompletionInference)

    # make the request
    body = {
        "model": "ggml-gpt4all-l13b-snoozy.bin",
        "prompt": "This is a test prompt"
    }

    response = client.post(
        "/aimodels/gpt4all/basic/completions",
        headers={},
        json=body,
    )

    inputs = CompletionInferenceInputs(model=Gpt4AllModelFilenameEnum.L13B_SNOOZY,
                              prompt='This is a test prompt',
                              max_tokens=256,
                              temperature=0.8,
                              top_p=0.95,
                              n=1,
                              stream=False,
                              echo=False,
                              stop=[],
                              presence_penalty=1.3
                              )

    # check that the mock was called with the correct arguments
    MockCompletionInferenceObject.basic_response.assert_called_once_with(inputs)

    # check that the response is correct
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "This is a test response"


def test_gpt_completion_post_model_invalid_gpt4all_model_type(client: TestClient):

    # make the request with an invalid model_type
    body = {
        "model": "invalid",
        "prompt": "This is a test prompt"
    }

    response = client.post(
        "/aimodels/gpt4all/basic/completions",
        headers={},
        json=body,
    )

    # check that the response is correct
    assert response.status_code == 422
    assert response.json()['detail'] is not None

def test_chat_completion_post_valid_input(client: TestClient, mocker):

    MockCompletionInferenceObject, MockCompletionInference = gen_mock_completion_inference(
        "This is a test response")

    # patch the CompletionInference class with the MockCompletionInference
    mocker.patch("app.aimodels.gpt4all.routers.completions.CompletionInference",
                 new=MockCompletionInference)

    # make the request
    body = {
        "model": "ggml-gpt4all-l13b-snoozy.bin",
        "prompt": "This is a test prompt"
    }

    response = client.post(
        "/aimodels/gpt4all/chat/completions",
        headers={},
        json=body,
    )

    inputs = CompletionInferenceInputs(model=Gpt4AllModelFilenameEnum.L13B_SNOOZY,
                              prompt='This is a test prompt',
                              max_tokens=256,
                              temperature=0.8,
                              top_p=0.95,
                              n=1,
                              stream=False,
                              echo=False,
                              stop=[],
                              presence_penalty=1.3
                              )

    # check that the mock was called with the correct arguments
    MockCompletionInferenceObject.chat_response.assert_called_once_with(inputs)

    # check that the response is correct
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "This is a test response"
