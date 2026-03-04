import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app  # Assuming the FastAPI app is initialized in `app.main`
from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInferenceInputs
from app.chat_search.ai_services.service import RetrievalService


def test_chat_query_retrieval_get_invalid_prompt(client: TestClient, mocker):
    # Mock the validate_inputs_and_generate_service to avoid actual dependencies
    mocker.patch(
        "app.aimodels.gpt4all.routers.completions.validate_inputs_and_generate_service",
        side_effect=ValueError("Invalid inputs"),
    )

    # Send a request with an invalid prompt (e.g., empty string)
    response = client.get("/retrieval", params={"prompt": ""})

    # Assert that the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422


def test_chat_query_retrieval_get_missing_dependencies(client: TestClient, mocker):
    # Mock the validate_inputs_and_generate_service to simulate missing dependencies
    mocker.patch(
        "app.aimodels.gpt4all.routers.completions.validate_inputs_and_generate_service",
        side_effect=Exception("Missing dependencies"),
    )

    # Send a request with a valid prompt but simulate missing dependencies
    response = client.get("/retrieval", params={"prompt": "test prompt"})

    # Assert that the response status code is 500 (Internal Server Error)
    assert response.status_code == 422


def test_chat_query_retrieval_get_template_not_found(client: TestClient, mocker):
    # Mock the RetrievalService to return a valid result
    mocker.patch(
        "app.aimodels.gpt4all.routers.completions.validate_inputs_and_generate_service",
        return_value=MagicMock(),
    )

    mocker.patch(
        "app.dependencies.get_db",
        return_value=MagicMock(),
    )

    mocker.patch(
        "app.dependencies.get_s3",
        return_value=MagicMock(),
    )

    mocker.patch(
        "app.chat_search.ai_services.service.RetrievalService.retrieve",
        return_value={"result": "test result"},
    )

    # # Mock the Jinja2 environment to raise a TemplateNotFound exception
    # mocker.patch(
    #     "app.chat_search.ai_services.service.Environment.get_template",
    #     side_effect=Exception("Template not found"),
    # )

    # Send a request with a valid prompt
    response = client.get("/retrieval", params={"prompt": "test prompt"})

    # Assert that the response status code is 500 (Internal Server Error)
    assert response.status_code == 422
