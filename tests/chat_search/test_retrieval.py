from unittest.mock import MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jinja2.exceptions import TemplateNotFound


def test_chat_query_retrieval_get_invalid_prompt(client: TestClient, mocker):
    # Mock the validate_inputs_and_generate_service to avoid actual dependencies
    mocker.patch(
        "app.chat_search.routers.retrieval.validate_inputs_and_generate_service",
        side_effect=HTTPException(
            status_code=422,
            detail="Invalid model type or no uploaded pretrained model for this type"),
    )

    # Send a request with an invalid prompt (e.g., empty string)
    response = client.get("/retrieval", params={"prompt": ""})

    # Assert that the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422


def test_chat_query_retrieval_get_missing_dependencies(client: TestClient, mocker):
    # Mock the validate_inputs_and_generate_service to simulate missing dependencies
    mocker.patch(
        "app.chat_search.routers.retrieval.validate_inputs_and_generate_service",
        side_effect=HTTPException(
            status_code=422,
            detail="Invalid model type or no uploaded pretrained model for this type"),
    )

    # Send a request with a valid prompt but simulate missing dependencies
    response = client.get("/retrieval", params={"prompt": "test prompt"})

    # Assert that the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422


def test_chat_query_retrieval_get_template_not_found(client: TestClient, mocker):
    # Mock the RetrievalService to return a valid result
    mocker.patch(
        "app.chat_search.routers.retrieval.validate_inputs_and_generate_service",
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

    mock_retrieval_service = MagicMock()
    mock_retrieval_service.retrieve.return_value = {"result": "test result"}

    mocker.patch(
        "app.chat_search.routers.retrieval.RetrievalService",
        return_value=mock_retrieval_service,
    )

    # Mock the HTML templating step for a TemplateNotFound exception
    mocker.patch(
        "app.chat_search.routers.retrieval._render_result_as_html",
        side_effect=TemplateNotFound("HTML template not found"),
    )

    # Send a request with a valid prompt
    response = client.get("/retrieval", params={"prompt": "test prompt"})

    assert response.status_code == 500
