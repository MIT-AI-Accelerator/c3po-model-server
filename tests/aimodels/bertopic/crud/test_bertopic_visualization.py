from sqlalchemy.orm import Session
from app.aimodels.bertopic.crud.crud_bertopic_visualization import bertopic_visualization
from app.ppg_common.schemas.bertopic.bertopic_visualization import  BertopicVisualizationTypeEnum


def test_get_bertopic_visualization_no_id(db: Session):
    result = bertopic_visualization.get_by_model_or_topic_id(
        db, model_or_topic_id='', visualization_type=BertopicVisualizationTypeEnum.MODEL_WORDS)
    assert result is None
