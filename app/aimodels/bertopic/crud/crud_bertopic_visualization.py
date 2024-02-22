from typing import Union
from sqlalchemy.orm import Session
from ppg.schemas.bertopic.bertopic_visualization import BertopicVisualizationCreate, BertopicVisualizationTypeEnum
from app.crud.base import CRUDBase
from ..models.bertopic_visualization import BertopicVisualizationModel


class CRUDBertopicVisualization(CRUDBase[BertopicVisualizationModel, BertopicVisualizationCreate, BertopicVisualizationCreate]):
    # CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])

    def get_by_model_or_topic_id(self, db: Session, *, model_or_topic_id: str, visualization_type: BertopicVisualizationTypeEnum) -> Union[BertopicVisualizationModel, None]:
        if not model_or_topic_id:
            return None

        return db.query(self.model).filter(self.model.model_or_topic_id == model_or_topic_id,
                                           self.model.visualization_type == visualization_type).first()


bertopic_visualization = CRUDBertopicVisualization(BertopicVisualizationModel)
