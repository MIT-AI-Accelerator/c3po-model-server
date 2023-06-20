from sqlalchemy.orm import Session
from app.aimodels.gpt4all import crud
from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInferenceInputs, CompletionInferenceOutputChoices, CompletionInferenceOutputs, FinishReasonEnum

from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllModelFilenameEnum, Gpt4AllPretrainedModel
from app.aimodels.gpt4all.schemas.gpt4all_pretrained import Gpt4AllPretrainedCreate

import pytest


@pytest.fixture(scope="function")
def gpt4all_pretrained_obj_uploaded_false(valid_sha256: str, db: Session) -> Gpt4AllPretrainedModel:
    gpt4all_pretrained_obj = Gpt4AllPretrainedCreate(sha256=valid_sha256)
    new_gpt4all_pretrained_obj: Gpt4AllPretrainedModel = crud.gpt4all_pretrained.create(
        db, obj_in=gpt4all_pretrained_obj)

    return new_gpt4all_pretrained_obj


@pytest.fixture(scope="function")
def gpt4all_pretrained_obj_uploaded_true(valid_sha256: str, db: Session) -> Gpt4AllPretrainedModel:
    gpt4all_pretrained_obj = Gpt4AllPretrainedCreate(sha256=valid_sha256)
    new_gpt4all_pretrained_obj: Gpt4AllPretrainedModel = crud.gpt4all_pretrained.create(
        db, obj_in=gpt4all_pretrained_obj)

    # update the document to change uploaded to True
    new_gpt4all_pretrained_obj: Gpt4AllPretrainedModel = crud.gpt4all_pretrained.update(
        db, db_obj=new_gpt4all_pretrained_obj, obj_in={"uploaded": True})

    return new_gpt4all_pretrained_obj


@pytest.fixture(scope="function")
def mock_completion_inference_outputs() -> CompletionInferenceOutputs:
    choices = CompletionInferenceOutputChoices(
        text="This is a test response",
        index=0,
        finish_reason=FinishReasonEnum.NULL
    )

    output_object = CompletionInferenceOutputs(
        model=Gpt4AllModelFilenameEnum.L13B_SNOOZY,
        choices=[choices]
    )

    return output_object

@pytest.fixture(scope="function")
def mock_completion_inference_inputs() -> CompletionInferenceInputs:
    return CompletionInferenceInputs(model=Gpt4AllModelFilenameEnum.L13B_SNOOZY,
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
