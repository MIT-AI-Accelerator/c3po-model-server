
from unittest.mock import patch, create_autospec
from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInference, CompletionInferenceInputs
from app.aimodels.gpt4all.models.llm_pretrained import LlmPretrainedModel
from minio import Minio
from pydantic import ValidationError

from langchain.llms.fake import FakeListLLM


@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=False)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.download_file_from_minio', return_value=True)
def test_create_completion_inference_object(mock_download_file_from_minio, mock_os_path, mock_llm_pretrained_obj):

    completion_inference_obj = CompletionInference(
        llm_pretrained_model_obj=mock_llm_pretrained_obj,
        s3=create_autospec(Minio)
    )

    mock_download_file_from_minio.assert_called_once()
    assert completion_inference_obj is not None

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=False)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.download_file_from_minio', return_value=True)
def test_create_completion_inference_object_fails_when_uploaded_false(mock_download_file_from_minio, mock_os_path, mock_llm_pretrained_obj):
    try:
        mock_llm_pretrained_obj.uploaded = False

        completion_inference_obj = CompletionInference(
            llm_pretrained_model_obj=mock_llm_pretrained_obj,
            s3=create_autospec(Minio)
        )

        assert False
    except ValidationError as e:
        mock_download_file_from_minio.assert_not_called()

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=False)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.download_file_from_minio', return_value=True)
def test_create_completion_inference_object_fails_when_model_type_none(mock_download_file_from_minio, mock_os_path, mock_llm_pretrained_obj: LlmPretrainedModel):
    try:
        mock_llm_pretrained_obj.model_type = None

        completion_inference_obj=CompletionInference(
            llm_pretrained_model_obj=mock_llm_pretrained_obj,
            s3=create_autospec(Minio)
        )

        assert False
    except ValidationError as e:
        mock_download_file_from_minio.assert_not_called()

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=True)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.GPT4All.__new__', return_value=FakeListLLM(responses=["test1", "test2"]))
def test_basic_response_lang_chain_works_with_fake_llm(mock_os_path, mock_gpt4all_new, mock_llm_pretrained_obj):

    completion_inference_obj = CompletionInference(
        llm_pretrained_model_obj=mock_llm_pretrained_obj,
        s3=create_autospec(Minio)
    )

    inputs = CompletionInferenceInputs(prompt="test")
    output = completion_inference_obj.basic_response(inputs)

    mock_gpt4all_new.assert_called_once()
    assert output.choices[0].text == "test1"

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=True)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.GPT4All.__new__', return_value=FakeListLLM(responses=["test1", "test2"]))
def test_question_response_lang_chain_works_with_fake_llm(mock_os_path, mock_gpt4all_new, mock_llm_pretrained_obj):

    completion_inference_obj = CompletionInference(
        llm_pretrained_model_obj=mock_llm_pretrained_obj,
        s3=create_autospec(Minio)
    )

    inputs = CompletionInferenceInputs(prompt="test",n=2)
    output = completion_inference_obj.basic_response(inputs)

    mock_gpt4all_new.assert_called_once()
    assert output.choices[0].text == "test1"
    assert output.choices[1].text == "test2"

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=True)
def test_type_validation_basic_and_question_response(mock_os_path, mock_llm_pretrained_obj: LlmPretrainedModel):
    completion_inference_obj=CompletionInference(
        llm_pretrained_model_obj=mock_llm_pretrained_obj,
        s3=create_autospec(Minio)
    )

    try:
        completion_inference_obj.basic_response(None)
        assert False
    except (ValidationError, TypeError) as e:
        assert True

    try:
        completion_inference_obj.question_response(None)
        assert False
    except (ValidationError, TypeError) as e:
        assert True
