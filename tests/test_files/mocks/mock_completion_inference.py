from time import time
from unittest.mock import MagicMock
from uuid import uuid4

from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInferenceOutputChoices, CompletionInferenceOutputs, FinishReasonEnum
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllModelFilenameEnum

def gen_mock_completion_inference(basic_response_return_value: str):
    """Generates a mock CompletionInference object and class that can be used to mock the
    CompletionInference class. This is used to mock the CompletionInference class in
    tests/aimodels/gpt4all/routers/test_completions.py

    Args:
        basic_response_return_value (str): The value that the mock CompletionInference
        object should return when the basic_response method is called

    Returns:
        A tuple containing the mock CompletionInference object and the mock
        CompletionInference class.

        MockCompletionInferenceObject (MagicMock): The mock CompletionInference object
        MockCompletionInference (MagicMock): The mock CompletionInference class
    """

    # Object that is created when CompletionInference is called
    MockCompletionInferenceObject = MagicMock()

    # output_object = {
    #     "id": str(uuid4()),
    #     "object": "text_completion",
    #     "created": int(time()),
    #     "model": "test-model",
    #     "choices": [
    #         {
    #         "text": f"{basic_response_return_value}",
    #         "index": 0,
    #         "logprobs": None,
    #         "finish_reason": "length"
    #         }
    #     ]
    # }

    choices = CompletionInferenceOutputChoices(
                text=f"{basic_response_return_value}",
                index=0,
                finish_reason=FinishReasonEnum.NULL
            )

    output_object = CompletionInferenceOutputs(
            model=Gpt4AllModelFilenameEnum.L13B_SNOOZY,
            choices=[choices]
        )

    # configure the mock to return the correct value for basic_response method
    attrs = {'basic_response.return_value': output_object, 'chat_response.return_value': output_object}
    MockCompletionInferenceObject.configure_mock(**attrs)

    # Mock the CompletionInference class, tell it to return the MockCompletionInferenceObject
    MockCompletionInference = MagicMock(
        return_value=MockCompletionInferenceObject)

    return MockCompletionInferenceObject, MockCompletionInference
