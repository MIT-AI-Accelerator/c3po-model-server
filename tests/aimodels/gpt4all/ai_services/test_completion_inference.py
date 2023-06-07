
from unittest.mock import patch, create_autospec
from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInference
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllPretrainedModel
from minio import Minio
from pydantic import ValidationError

from langchain.llms.fake import FakeListLLM


@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=False)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.download_file_from_minio', return_value=True)
def test_create_completion_inference_object(mock_download_file_from_minio, mock_os_path, mock_gpt4all_pretrained_obj):

    completion_inference_obj = CompletionInference(
        gpt4all_pretrained_model_obj=mock_gpt4all_pretrained_obj,
        s3=create_autospec(Minio)
    )

    mock_download_file_from_minio.assert_called_once()
    assert completion_inference_obj is not None

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=False)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.download_file_from_minio', return_value=True)
def test_create_completion_inference_object_fails_when_uploaded_false(mock_download_file_from_minio, mock_os_path, mock_gpt4all_pretrained_obj):
    try:
        mock_gpt4all_pretrained_obj.uploaded = False

        completion_inference_obj = CompletionInference(
            gpt4all_pretrained_model_obj=mock_gpt4all_pretrained_obj,
            s3=create_autospec(Minio)
        )

        assert False
    except ValidationError as e:
        mock_download_file_from_minio.assert_not_called()

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=False)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.download_file_from_minio', return_value=True)
def test_create_completion_inference_object_fails_when_model_type_none(mock_download_file_from_minio, mock_os_path, mock_gpt4all_pretrained_obj: Gpt4AllPretrainedModel):
    try:
        mock_gpt4all_pretrained_obj.model_type = None

        completion_inference_obj=CompletionInference(
            gpt4all_pretrained_model_obj=mock_gpt4all_pretrained_obj,
            s3=create_autospec(Minio)
        )

        assert False
    except ValidationError as e:
        mock_download_file_from_minio.assert_not_called()

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=True)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.GPT4All.__new__', return_value=FakeListLLM(responses=["test1", "test2"]))
def test_basic_response_lang_chain_works_with_fake_llm(mock_os_path, mock_gpt4all_new, mock_gpt4all_pretrained_obj):

    completion_inference_obj = CompletionInference(
        gpt4all_pretrained_model_obj=mock_gpt4all_pretrained_obj,
        s3=create_autospec(Minio)
    )

    output = completion_inference_obj.basic_response("test")
    mock_gpt4all_new.assert_called_once()
    assert output.completion == "test1"

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=True)
@patch('app.aimodels.gpt4all.ai_services.completion_inference.GPT4All.__new__', return_value=FakeListLLM(responses=["test1", "test2"]))
def test_question_response_lang_chain_works_with_fake_llm(mock_os_path, mock_gpt4all_new, mock_gpt4all_pretrained_obj):

    completion_inference_obj = CompletionInference(
        gpt4all_pretrained_model_obj=mock_gpt4all_pretrained_obj,
        s3=create_autospec(Minio)
    )

    output = completion_inference_obj.question_response("test")
    mock_gpt4all_new.assert_called_once()
    assert output.completion == "test1"

@patch('app.aimodels.gpt4all.ai_services.completion_inference.os.path.isfile', return_value=True)
def test_type_validation_basic_and_question_response(mock_os_path, mock_gpt4all_pretrained_obj: Gpt4AllPretrainedModel):
    completion_inference_obj=CompletionInference(
        gpt4all_pretrained_model_obj=mock_gpt4all_pretrained_obj,
        s3=create_autospec(Minio)
    )

    try:
        completion_inference_obj.basic_response(None)
        assert False
    except ValidationError as e:
        assert True

    try:
        completion_inference_obj.question_response(None)
        assert False
    except ValidationError as e:
        assert True
