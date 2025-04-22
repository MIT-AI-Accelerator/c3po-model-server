import pytest
from typing import Optional, Callable
from sqlalchemy.orm import Session

from app.core.config import settings
from app.main import app
from app.ppg_common.schemas.gpt4all.llm_pretrained import LlmPretrainedCreate
from app.aimodels.gpt4all import crud
from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInferenceInputs, CompletionInferenceOutputChoices, CompletionInferenceOutputs, FinishReasonEnum
from app.aimodels.gpt4all.models.llm_pretrained import LlmFilenameEnum, LlmPretrainedModel
from app.aimodels.bertopic.routers.bertopic_embedding_pretrained import get_db, get_s3

def setup(db, mock_s3):
    def replace_db():
        return db

    def mock_get_minio():
        return mock_s3

    app.dependency_overrides = {get_db: replace_db, get_s3: mock_get_minio}


def teardown():
    app.dependency_overrides = {}

@pytest.fixture(scope="function", autouse=True)
def setup_teardown(db, mock_s3):
    setup(db, mock_s3)
    yield
    teardown()

@pytest.fixture(scope="function")
def llm_pretrained_obj_uploaded_false(valid_sha256: str, db: Session) -> LlmPretrainedModel:
    llm_pretrained_obj = LlmPretrainedCreate(sha256=valid_sha256)
    new_llm_pretrained_obj: LlmPretrainedModel = crud.llm_pretrained.create(
        db, obj_in=llm_pretrained_obj)

    return new_llm_pretrained_obj


@pytest.fixture(scope="function")
def llm_pretrained_obj_uploaded_true(valid_sha256: str, db: Session) -> LlmPretrainedModel:
    llm_pretrained_obj = LlmPretrainedCreate(sha256=valid_sha256)
    new_llm_pretrained_obj: LlmPretrainedModel = crud.llm_pretrained.create(
        db, obj_in=llm_pretrained_obj)

    # update the document to change uploaded to True
    new_llm_pretrained_obj: LlmPretrainedModel = crud.llm_pretrained.update(
        db, db_obj=new_llm_pretrained_obj, obj_in={"uploaded": True})

    return new_llm_pretrained_obj


@pytest.fixture(scope="function")
def mock_completion_inference_outputs() -> CompletionInferenceOutputs:
    choices = CompletionInferenceOutputChoices(
        text="This is a test response",
        index=0,
        finish_reason=FinishReasonEnum.NULL
    )

    output_object = CompletionInferenceOutputs(
        model=LlmFilenameEnum.L13B_SNOOZY,
        choices=[choices]
    )

    return output_object

@pytest.fixture(scope="function")
def mock_completion_inference_inputs() -> CompletionInferenceInputs:
    return CompletionInferenceInputs(model=LlmFilenameEnum.L13B_SNOOZY,
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

@pytest.fixture(scope="function")
def mock_gpt4all_crud_sha256_return_vals() -> Callable:
    def side_effect(*args, **kwargs) -> Optional[LlmPretrainedModel]:
        '''
        return a nonempty object when called with the specific hash, else return None
        '''

        desired_input = settings.default_sha256_l13b_snoozy
        if args[0] == desired_input or kwargs["sha256"] == desired_input:
            return LlmPretrainedModel(uploaded=True)

        return None

    return side_effect
