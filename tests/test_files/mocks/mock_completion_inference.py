from unittest.mock import MagicMock

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

    # configure the mock to return the correct value for basic_response method
    attrs = {'basic_response.return_value': {
        'completion': basic_response_return_value}}
    MockCompletionInferenceObject.configure_mock(**attrs)

    # Mock the CompletionInference class, tell it to return the MockCompletionInferenceObject
    MockCompletionInference = MagicMock(
        return_value=MockCompletionInferenceObject)

    return MockCompletionInferenceObject, MockCompletionInference
