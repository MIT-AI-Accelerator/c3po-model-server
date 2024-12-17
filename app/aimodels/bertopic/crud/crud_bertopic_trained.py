from typing import Union
from pydantic import UUID4, BaseModel, ConfigDict
from datetime import datetime

from fastapi.encoders import jsonable_encoder
from sqlalchemy import desc
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import OriginationEnum
from app.crud.base import CRUDBase
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel
from app.ppg_common.schemas.bertopic.bertopic_trained import BertopicTrainedCreate, BertopicTrainedUpdate

class BertopicTrainedModelSummary(BaseModel):
    time: datetime
    id: UUID4
    sentence_transformer_id: UUID4
    weak_learner_id: Union[UUID4, None]
    summarization_model_id: Union[UUID4, None]

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, trained_model: BertopicTrainedModel):
        super().__init__(time = trained_model.time,
                         id = trained_model.id,
                         sentence_transformer_id = trained_model.sentence_transformer_id,
                         weak_learner_id = trained_model.weak_learner_id,
                         summarization_model_id = trained_model.summarization_model_id)

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDBertopicTrained(CRUDBase[BertopicTrainedModel, BertopicTrainedCreate, BertopicTrainedUpdate]  ):
    def create_with_embedding_pretrained_id(
        self, db: Session, *, obj_in: BertopicTrainedCreate, embedding_pretrained_id: UUID4
    ) -> Union[BertopicTrainedModel, None]:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data, embedding_pretrained_id=embedding_pretrained_id)
        db.add(db_obj)
        try:
            db.commit()
        except IntegrityError: #embedding_pretrained_id is not in db
            db.rollback()
            return None
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_embedding_pretrained_id(
        self, db: Session, *, embedding_pretrained_id: UUID4, skip: int = 0, limit: int = 100
    ) -> list[BertopicTrainedModel]:
        return (
            db.query(self.model)
            .filter(BertopicTrainedModel.embedding_pretrained_id == embedding_pretrained_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_trained_models(self, db: Session, *, row_limit = 1, originated_from = OriginationEnum.ORIGINATED_FROM_APP) -> list[BertopicTrainedModelSummary]:

        db_objs = db.query(self.model).filter(self.model.originated_from == originated_from,
                                              self.model.uploaded == True).order_by(desc(self.model.time)).limit(row_limit)
        return [BertopicTrainedModelSummary(trained_model = db_obj) for db_obj in db_objs]


bertopic_trained = CRUDBertopicTrained(BertopicTrainedModel)
