
from typing import Union
from sqlalchemy.orm import Session
from ppg.config import OriginationEnum
from app.crud.base import CRUDBase
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from ..schemas.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDBertopicEmbeddingPretrained(CRUDBase[BertopicEmbeddingPretrainedModel, BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate]):
    def get_by_sha256(self, db: Session, *, sha256: str) -> Union[BertopicEmbeddingPretrainedModel, None]:
        if not sha256:
            return None

        return db.query(self.model).filter(self.model.sha256 == sha256).first()

    def get_by_model_name(self, db: Session, *, model_name: str, originated_from = OriginationEnum.ORIGINATED_FROM_APP) -> Union[BertopicEmbeddingPretrainedModel, None]:
        if not model_name:
            return None

        return db.query(self.model).filter(self.model.model_name == model_name,
                                           self.model.uploaded,
                                           self.model.originated_from == originated_from).order_by(self.model.version.desc()).first()


bertopic_embedding_pretrained = CRUDBertopicEmbeddingPretrained(
    BertopicEmbeddingPretrainedModel)
