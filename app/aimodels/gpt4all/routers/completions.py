from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter
from minio import Minio
from app.dependencies import get_db, get_minio
from sqlalchemy.orm import Session
from .. import crud
from app.core.errors import ValidationError, HTTPValidationError
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
    response_model=Union[OutputResponse, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="GPT completion endpoint",
    response_description="Completed GPT response"
)
def gpt_completion_post(request: InputRequest, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    Union[OutputResponse, HTTPValidationError]
):
    """
    GPT completion endpoint.
    """
    # check to make sure id exists
    gpt4all_pretrained_model_obj: Gpt4AllPretrainedModel = crud.gpt4all_pretrained.get(
        db, request.gpt4all_pretrained_id)
    if not gpt4all_pretrained_model_obj:
        return HTTPValidationError(detail=[ValidationError(loc=['path', 'gpt4all_pretrained model upload'], msg='Invalid pretrained model id', type='value_error')])

    # check to make sure gpt4all_pretrained_model_obj is uploaded
    if not gpt4all_pretrained_model_obj.uploaded:
        return HTTPValidationError(detail=[ValidationError(loc=['path', 'gpt4all_pretrained upload'], msg='gpt4all_pretrained model type has not been uploaded', type='value_error')])

    completion_inference_service = CompletionInference(gpt4all_pretrained_model_obj=gpt4all_pretrained_model_obj, s3=s3)
    output = completion_inference_service.basic_response(request.prompt)

    return output
