from fastapi import Depends, APIRouter, HTTPException
from minio import Minio
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllModelFilenameEnum, Gpt4AllPretrainedModel
from app.aimodels.gpt4all.routers.completions import validate_inputs_and_generate_service
from app.dependencies import get_db, get_minio
from app.core.config import settings
from sqlalchemy.orm import Session
from .. import crud
from app.aimodels.gpt4all.ai_services.completion_inference import CompletionInference, CompletionInferenceInputs, CompletionInferenceOutputs
from .service import RetrievalService

router = APIRouter()

@router.post(
    "/retrieval",
    response_model={},
    summary="Query retrieval endpoint",
    response_description="Answerwed query with sources"
)
async def chat_query_retrieval_post(request: CompletionInferenceInputs, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    CompletionInferenceOutputs
):
    """
    Query retrieval endpoint.
    """

    completion_inference_service: CompletionInference = validate_inputs_and_generate_service(request, db, s3)
    retrieval_service = RetrievalService(completion_inference_service)
    retrieval_service.question_response(request)
    return {}
