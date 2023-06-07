from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
from minio import Minio
from app.dependencies import get_db, get_minio
from sqlalchemy.orm import Session
from .. import crud
from app.core.model_cache import MODEL_CACHE_BASEDIR
from ..models import Gpt4AllPretrainedModel
from ..ai_services.completion_inference import CompletionInference

router = APIRouter(
    prefix="/completions"
)

class InputRequest(BaseModel):
    gpt4all_pretrained_id: UUID4
    prompt: str


class OutputResponse(BaseModel):
    completion: str


@router.post(
    "/",
    response_model=OutputResponse,
    summary="GPT completion endpoint",
    response_description="Completed GPT response"
)
def gpt_completion_post(request: InputRequest, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    OutputResponse
):
    """
    GPT completion endpoint.
    """

    # check to make sure id exists
    gpt4all_pretrained_model_obj: Gpt4AllPretrainedModel = crud.gpt4all_pretrained.get(
        db, request.gpt4all_pretrained_id)
    if not gpt4all_pretrained_model_obj:
        raise HTTPException(
            status_code=422, detail="Invalid pretrained model id")

    # check to make sure gpt4all_pretrained_model_obj is uploaded
    if not gpt4all_pretrained_model_obj.uploaded:
        raise HTTPException(
            status_code=422, detail="gpt4all_pretrained upload")

    completion_inference_service = CompletionInference(
        gpt4all_pretrained_model_obj=gpt4all_pretrained_model_obj, s3=s3)
    output = completion_inference_service.basic_response(request.prompt)

    return output
