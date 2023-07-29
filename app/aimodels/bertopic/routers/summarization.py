from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.core.errors import ValidationError, HTTPValidationError
# from ..models.topic import TopicDocument

router = APIRouter(
    prefix=""
)


class SummarizationRequest(BaseModel):
    trained_topic_id: UUID4


class SummarizationResponse(BaseModel):
    # documents: list[TopicDocument]
    summary: str


@router.get(
    "/topic/{id}",
    response_model=Union[SummarizationResponse, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve a BERTopic summary",
    response_description="Retrieved a BERTopic summary")
async def get_topic_summary(id: UUID4, db: Session = Depends(get_db)) -> (
    Union[SummarizationResponse, HTTPValidationError]
):
    """
    Retrieve a BERTopic summary and list of most relaetd documents

    - **trained_topic_id**: Required.  Trained BERTopic model topic ID.
    """

    # response_obj = SummarizationResponse(
    #     documents=[TopicDocument(text="cats", relation=100.0), TopicDocument(text="dogs", relation=0.0)], summary="i like cats")
    response_obj = SummarizationResponse(summary="i like cats")

    return response_obj
