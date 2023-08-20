from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.core.errors import ValidationError, HTTPValidationError
from ..schemas.topic import TopicSummary
from .. import crud

router = APIRouter(
    prefix=""
)


@router.get(
    "/topic/{id}",
    response_model=Union[TopicSummary, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve a BERTopic summary",
    response_description="Retrieved a BERTopic summary")
async def get_topic_summary(id: UUID4, db: Session = Depends(get_db)) -> (
    Union[TopicSummary, HTTPValidationError]
):
    """
    Retrieve a BERTopic summary and list of most relaetd documents

    - **trained_topic_id**: Required.  Trained BERTopic model topic ID.
    """
    topic_obj = crud.topic_summary.get(db, id)
    if not topic_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic topic summary not found")

    return topic_obj

@router.get(
    "/topic/{id}/visualize_topic_timeline",
    response_class=HTMLResponse,
    summary="Retrieve a BERTopic timeline visualization",
    response_description="Retrieved a BERTopic timeline visualization")
async def visualize_topic_timeline(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic timeline visualization

    - **trained_topic_id**: Required.  Trained BERTopic model ID.
    """

    topic_obj = crud.topic_summary.get(db, id)
    if not topic_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic topic summary not found")

    return topic_obj.topic_timeline_visualization
