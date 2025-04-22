from fastapi import Depends, APIRouter, HTTPException
from app.aimodels.gpt4all.models.llm_pretrained import LlmFilenameEnum
from app.dependencies import get_db, get_s3
from app.core.config import settings
from sqlalchemy.orm import Session
from mypy_boto3_s3.client import S3Client
from .. import crud
from ..models import LlmPretrainedModel
from ..ai_services.completion_inference import CompletionInference, CompletionInferenceInputs, CompletionInferenceOutputs

router = APIRouter()

@router.post(
    "/basic/completions",
    response_model=CompletionInferenceOutputs,
    summary="GPT completion endpoint",
    response_description="Completed GPT response"
)
async def gpt_completion_post(request: CompletionInferenceInputs, db: Session = Depends(get_db), s3: S3Client = Depends(get_s3)) -> (
    CompletionInferenceOutputs
):
    """
    GPT completion endpoint.
    """

    completion_inference_service = validate_inputs_and_generate_service(request, db, s3)
    return completion_inference_service.basic_response(request)

@router.post(
    "/chat/completions",
    response_model=CompletionInferenceOutputs,
    summary="Chat completion endpoint",
    response_description="Completed Chat response"
)
async def chat_completion_post(request: CompletionInferenceInputs, db: Session = Depends(get_db), s3: S3Client = Depends(get_s3)) -> (
    CompletionInferenceOutputs
):
    """
    Chat completion endpoint.
    """

    completion_inference_service = validate_inputs_and_generate_service(request, db, s3)
    return completion_inference_service.chat_response(request)

def validate_inputs_and_generate_service(request: CompletionInferenceInputs, db: Session, s3: S3Client):
    # default model to pull
    sha256 = settings.default_sha256_l13b_snoozy

    # check for other model type options
    if request.model == LlmFilenameEnum.L13B_SNOOZY:
        sha256 = settings.default_sha256_l13b_snoozy

    # TODO: determine how to incorporate crud.llm_pretrained.get_latest_uploaded_by_model_type
    llm_pretrained_model_obj: LlmPretrainedModel = crud.llm_pretrained.get_by_sha256(
        db, sha256=sha256)

    if not llm_pretrained_model_obj or not llm_pretrained_model_obj.uploaded:
        raise HTTPException(
            status_code=422, detail="Invalid model type or no uploaded pretrained model for this type")

    return CompletionInference(
        llm_pretrained_model_obj=llm_pretrained_model_obj, s3=s3)
