from pydantic import UUID4

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.crud.base import CRUDBase
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel
from app.aimodels.bertopic.schemas.bertopic_trained import BertopicTrainedCreate, BertopicTrainedUpdate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDBertopicTrained(CRUDBase[BertopicTrainedModel, BertopicTrainedCreate, BertopicTrainedUpdate]  ):
    def create_with_embedding_pretrained_id(
        self, db: Session, *, obj_in: BertopicTrainedCreate, embedding_pretrained_id: UUID4
    ) -> BertopicTrainedModel:
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

bertopic_trained = CRUDBertopicTrained(BertopicTrainedModel)
