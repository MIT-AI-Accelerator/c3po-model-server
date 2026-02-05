from unittest.mock import create_autospec
from sqlalchemy.orm import Session

from app.aimodels.gpt4all.models.llm_pretrained import LlmPretrainedModel

import pytest

@pytest.fixture(scope="function")
def mock_llm_pretrained_obj(valid_sha256: str, db: Session) -> LlmPretrainedModel:
    # need to set any attributes that could be validated and only do one wrong at a time
    llm_pretrained_obj = create_autospec(LlmPretrainedModel)
    llm_pretrained_obj.model_type = "ggml-gpt4all-l13b-snoozy.bin"
    llm_pretrained_obj.uploaded = True

    return llm_pretrained_obj
