from fastapi import Depends, APIRouter, HTTPException
from minio import Minio
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllModelFilenameEnum
from app.dependencies import get_db, get_minio
from sqlalchemy.orm import Session
from .. import crud
from ..models import Gpt4AllPretrainedModel
from ..ai_services.completion_inference import CompletionInference, CompletionInferenceInputs, CompletionInferenceOutputs

router = APIRouter()

@router.post(
    "/basic/completions",
    response_model=CompletionInferenceOutputs,
    summary="GPT completion endpoint",
    response_description="Completed GPT response"
)
def gpt_completion_post(request: CompletionInferenceInputs, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    CompletionInferenceOutputs
):
    """
    GPT completion endpoint.
    """

    completion_inference_service = _validate_inputs_and_generate_service(request, db, s3)
    return completion_inference_service.basic_response(request)

@router.post(
    "/chat/completions",
    response_model=CompletionInferenceOutputs,
    summary="Chat completion endpoint",
    response_description="Completed Chat response"
)
def chat_completion_post(request: CompletionInferenceInputs, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    CompletionInferenceOutputs
):
    """
    Chat completion endpoint.
    """

    completion_inference_service = _validate_inputs_and_generate_service(request, db, s3)
    return completion_inference_service.chat_response(request)

def _validate_inputs_and_generate_service(request: CompletionInferenceInputs, db: Session, s3: Minio):
    sha256 = "997072bd77078c82131e7becf3fc4b090efec43a1f480bbde0e401ffe5145688"
    if request.model == Gpt4AllModelFilenameEnum.L13B_SNOOZY:
        sha256 = "997072bd77078c82131e7becf3fc4b090efec43a1f480bbde0e401ffe5145688"

    # TODO: determine how to incorporate crud.gpt4all_pretrained.get_latest_uploaded_by_model_type
    gpt4all_pretrained_model_obj: Gpt4AllPretrainedModel = crud.gpt4all_pretrained.get_by_sha256(
        db, sha256=sha256)

    if not gpt4all_pretrained_model_obj or not gpt4all_pretrained_model_obj.uploaded:
        raise HTTPException(
            status_code=422, detail="Invalid model type or no uploaded pretrained model for this type")

    return CompletionInference(
        gpt4all_pretrained_model_obj=gpt4all_pretrained_model_obj, s3=s3)
