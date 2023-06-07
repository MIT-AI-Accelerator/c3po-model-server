from fastapi.testclient import TestClient
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllPretrainedModel

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


def test_gpt_completion_post_valid_input(client: TestClient, gpt4all_pretrained_obj_uploaded_true: Gpt4AllPretrainedModel, mocker):

    MockCompletionInferenceObject, MockCompletionInference = gen_mock_completion_inference(
        "This is a test response")

    # patch the CompletionInference class with the MockCompletionInference
    mocker.patch("app.aimodels.gpt4all.routers.completions.CompletionInference",
                 new=MockCompletionInference)

    # make the request
    body = {
        "gpt4all_pretrained_id": str(gpt4all_pretrained_obj_uploaded_true.id),
        "prompt": "This is a test prompt",
    }

    response = client.post(
        "/aimodels/gpt4all/completions",
        headers={},
        json=body,
    )

    # check that the mock was called with the correct arguments
    MockCompletionInferenceObject.basic_response.assert_called_once_with(
        "This is a test prompt")

    # check that the response is correct
    assert response.status_code == 200
    assert response.json() == {"completion": "This is a test response"}

def test_gpt_completion_post_model_not_uploaded(client: TestClient, gpt4all_pretrained_obj_uploaded_false: Gpt4AllPretrainedModel, mocker):

    MockCompletionInferenceObject, MockCompletionInference = gen_mock_completion_inference(
        "This is a test response")

    # patch the CompletionInference class with the MockCompletionInference
    mocker.patch("app.aimodels.gpt4all.routers.completions.CompletionInference",
                 new=MockCompletionInference)

    # make the request
    body = {
        "gpt4all_pretrained_id": str(gpt4all_pretrained_obj_uploaded_false.id),
        "prompt": "This is a test prompt",
    }

    response = client.post(
        "/aimodels/gpt4all/completions",
        headers={},
        json=body,
    )

    # check that the mock was called with the correct arguments
    MockCompletionInferenceObject.basic_response.assert_not_called()

    # check that the response is correct
    assert response.status_code == 422
    assert response.json()['detail'] is not None


def test_gpt_completion_post_model_invalid_gpt4all_model_id(client: TestClient):

    # make the request with an invalid gpt4all_pretrained_id
    body = {
        "gpt4all_pretrained_id": "bcf3a0e0-0b1a-3e8e-9b0a-9e8b9b9b9b9b",
        "prompt": "This is a test prompt",
    }

    response = client.post(
        "/aimodels/gpt4all/completions",
        headers={},
        json=body,
    )

    # check that the response is correct
    assert response.status_code == 422
    assert response.json()['detail'] is not None
