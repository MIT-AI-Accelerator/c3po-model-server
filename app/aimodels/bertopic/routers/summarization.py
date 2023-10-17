import json
from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
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
    "/model/{id}/visualize_topic_clusters",
    response_class=HTMLResponse,
    summary="Retrieve a BERTopic document cluster visualization",
    response_description="Retrieved a BERTopic document cluster visualization")
async def visualize_topic_clusters(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic document cluster visualization

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """

    model_obj = crud.bertopic_trained.get(db, id)
    if not model_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic trained model not found")

    return model_obj.topic_cluster_visualization


@router.get(
    "/model/{id}/visualize_topic_words",
    response_class=HTMLResponse,
    summary="Retrieve a BERTopic word probability visualization",
    response_description="Retrieved a BERTopic word probability visualization")
async def visualize_topic_words(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic word probability visualization

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """

    model_obj = crud.bertopic_trained.get(db, id)
    if not model_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic trained model not found")

    return model_obj.topic_word_visualization


@router.get(
    "/model/{id}/topics",
    response_model=Union[list[TopicSummary], HTTPValidationError],
    summary="Retrieve topics for a trained BERTopic model",
    response_description="Retrieved topics for a trained BERTopic model")
async def get_model_topics(id: UUID4, db: Session = Depends(get_db)) -> (
    Union[list[TopicSummary], HTTPValidationError]
):
    """
    Retrieve topics for a trained BERTopic model

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """

    if not crud.bertopic_trained.get(db, id):
        raise HTTPException(
            status_code=422, detail=f"BERTopic trained model not found")

    return crud.topic_summary.get_by_model_id(db, model_id=id)

@router.get(
    "/topic/{id}",
    response_model=Union[str, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Retrieve a BERTopic summary",
    response_description="Retrieved a BERTopic summary")
async def get_topic_summary(id: UUID4, db: Session = Depends(get_db)) -> (
    Union[str, HTTPValidationError]
):
    """
    Retrieve a BERTopic summary and list of most relaetd documents

    - **trained_topic_id**: Required.  Trained BERTopic model topic ID.
    """
    topic_obj = crud.topic_summary.get(db, id)
    if not topic_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic topic summary not found")

    # drop topic_timeline_visualization from topic summary
    json_obj = jsonable_encoder(topic_obj)
    json_obj.pop('topic_timeline_visualization')

    return json.dumps(json_obj, indent=2)

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
