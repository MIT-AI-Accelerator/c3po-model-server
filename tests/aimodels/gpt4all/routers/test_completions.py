from typing import Callable
from unittest.mock import create_autospec
from fastapi.testclient import TestClient
from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInference, CompletionInferenceInputs, CompletionInferenceOutputs
from app.aimodels.gpt4all import crud as crud_gpt4all

def test_gpt_completion_post_valid_input(client: TestClient,
                                         mock_completion_inference_inputs: CompletionInferenceInputs,
                                         mock_completion_inference_outputs: CompletionInferenceOutputs,
                                         mock_gpt4all_crud_sha256_return_vals: Callable,
                                         mocker):

    # mock the return value
    mock_crud = create_autospec(crud_gpt4all)
    mocker.patch("app.aimodels.gpt4all.routers.completions.crud", new=mock_crud)
    mock_crud.llm_pretrained.get_by_sha256.side_effect = mock_gpt4all_crud_sha256_return_vals

    # mock the CompletionInference Object
    mock_completion_inference_object = create_autospec(CompletionInference)
    mocker.patch("app.aimodels.gpt4all.routers.completions.CompletionInference.__new__",
                 return_value=mock_completion_inference_object)

    mock_completion_inference_object.basic_response.return_value = mock_completion_inference_outputs

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

    # check that the mock was called with the correct arguments
    mock_completion_inference_object.basic_response.assert_called_once_with(mock_completion_inference_inputs)

    # check that the response is correct
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "This is a test response"


def test_gpt_completion_post_model_invalid_llm_model_type(client: TestClient):

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

def test_chat_completion_post_valid_input(client: TestClient,
                                         mock_completion_inference_inputs: CompletionInferenceInputs,
                                         mock_completion_inference_outputs: CompletionInferenceOutputs,
                                         mock_gpt4all_crud_sha256_return_vals: Callable,
                                         mocker):

    # mock the return value
    mock_crud = create_autospec(crud_gpt4all)
    mocker.patch("app.aimodels.gpt4all.routers.completions.crud", new=mock_crud)
    mock_crud.llm_pretrained.get_by_sha256.side_effect = mock_gpt4all_crud_sha256_return_vals

    # mock the CompletionInference Object
    mock_completion_inference_object = create_autospec(CompletionInference)
    mocker.patch("app.aimodels.gpt4all.routers.completions.CompletionInference.__new__",
                 return_value=mock_completion_inference_object)

    mock_completion_inference_object.chat_response.return_value = mock_completion_inference_outputs

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

    # check that the mock was called with the correct arguments
    mock_completion_inference_object.chat_response.assert_called_once_with(mock_completion_inference_inputs)

    # check that the response is correct
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "This is a test response"
