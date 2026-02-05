from typing import Union, Any
from pydantic import UUID4
from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.core.errors import HTTPValidationError
from app.ppg_common.schemas.bertopic.topic import TopicSummary
from ..models.bertopic_visualization import BertopicVisualizationTypeEnum
from .. import crud

router = APIRouter(
    prefix=""
)

@router.get(
    "/models",
    response_model=Union[list[Any], HTTPValidationError],
    summary="Retrieve all available trained BERTopic models",
    response_description="Retrieved trained BERTopic models")
async def get_model_topics(limit: int = 1, db: Session = Depends(get_db)) -> (
    Union[list[Any], HTTPValidationError]):
    """
    Retrieve all available trained BERTopic models

    - **limit**: Optional.  Maximum number of trained model IDs to return.
    """
    return crud.bertopic_trained.get_trained_models(db, row_limit=limit)

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
    vis_obj = crud.bertopic_visualization.get_by_model_or_topic_id(db,
                                                                   model_or_topic_id=id,
                                                                   visualization_type=BertopicVisualizationTypeEnum.MODEL_CLUSTERS)
    if not vis_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic visualization not found")

    return vis_obj.html_string


@router.get(
    "/model/{id}/visualize_topic_clusters/json",
    response_model=Union[str, HTTPValidationError],
    summary="Retrieve a BERTopic document cluster visualization",
    response_description="Retrieved a BERTopic document cluster visualization")
async def visualize_topic_clusters(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic document cluster visualization

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """
    vis_obj = crud.bertopic_visualization.get_by_model_or_topic_id(db,
                                                                   model_or_topic_id=id,
                                                                   visualization_type=BertopicVisualizationTypeEnum.MODEL_CLUSTERS)
    if not vis_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic visualization not found")

    return vis_obj.json_string


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
    vis_obj = crud.bertopic_visualization.get_by_model_or_topic_id(db,
                                                                   model_or_topic_id=id,
                                                                   visualization_type=BertopicVisualizationTypeEnum.MODEL_WORDS)
    if not vis_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic visualization not found")

    return vis_obj.html_string


@router.get(
    "/model/{id}/visualize_topic_words/json",
    response_model=Union[str, HTTPValidationError],
    summary="Retrieve a BERTopic word probability visualization",
    response_description="Retrieved a BERTopic word probability visualization")
async def visualize_topic_words(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic word probability visualization

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """
    vis_obj = crud.bertopic_visualization.get_by_model_or_topic_id(db,
                                                                   model_or_topic_id=id,
                                                                   visualization_type=BertopicVisualizationTypeEnum.MODEL_WORDS)
    if not vis_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic visualization not found")

    return vis_obj.json_string


@router.get(
    "/model/{id}/visualize_topic_timeline",
    response_class=HTMLResponse,
    summary="Retrieve a BERTopic model-level topic timeline visualization",
    response_description="Retrieved a BERTopic model-level topic timeline visualization")
async def visualize_topic_words(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic model-level topic timeline visualization

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """
    vis_obj = crud.bertopic_visualization.get_by_model_or_topic_id(db,
                                                                   model_or_topic_id=id,
                                                                   visualization_type=BertopicVisualizationTypeEnum.MODEL_TIMELINE)
    if not vis_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic visualization not found")

    return vis_obj.html_string


@router.get(
    "/model/{id}/visualize_topic_timeline/json",
    response_model=Union[str, HTTPValidationError],
    summary="Retrieve a BERTopic model-level topic timeline visualization",
    response_description="Retrieved a BERTopic model-level topic timeline visualization")
async def visualize_topic_words(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic model-level topic timeline visualization

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """
    vis_obj = crud.bertopic_visualization.get_by_model_or_topic_id(db,
                                                                   model_or_topic_id=id,
                                                                   visualization_type=BertopicVisualizationTypeEnum.MODEL_TIMELINE)
    if not vis_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic visualization not found")

    return vis_obj.json_string


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
    vis_obj = crud.bertopic_visualization.get_by_model_or_topic_id(db,
                                                                   model_or_topic_id=id,
                                                                   visualization_type=BertopicVisualizationTypeEnum.TOPIC_TIMELINE)
    if not vis_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic visualization not found")

    return vis_obj.html_string


@router.get(
    "/topic/{id}/visualize_topic_timeline/json",
    response_model=Union[str, HTTPValidationError],
    summary="Retrieve a BERTopic timeline visualization",
    response_description="Retrieved a BERTopic timeline visualization")
async def visualize_topic_timeline(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic timeline visualization

    - **trained_topic_id**: Required.  Trained BERTopic model ID.
    """
    vis_obj = crud.bertopic_visualization.get_by_model_or_topic_id(db,
                                                                   model_or_topic_id=id,
                                                                   visualization_type=BertopicVisualizationTypeEnum.TOPIC_TIMELINE)
    if not vis_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic visualization not found")

    return vis_obj.json_string
