from typing import Union
from sqlalchemy.orm import Session
from app.crud.base import CRUDBase
from ..models.topic import TopicSummaryModel
from ..schemas.topic import TopicSummaryCreate


class CRUDTopicSummary(CRUDBase[TopicSummaryModel, TopicSummaryCreate, TopicSummaryCreate]):
    # CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])

    def get_by_model_id(self, db: Session, *, model_id: str) -> Union[TopicSummaryModel, None]:
        if not model_id:
            return None

        return db.query(self.model).filter(self.model.model_id == model_id).all()


topic_summary = CRUDTopicSummary(TopicSummaryModel)
