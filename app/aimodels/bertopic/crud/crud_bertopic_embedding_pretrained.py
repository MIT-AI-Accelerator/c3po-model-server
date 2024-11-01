
from typing import Union
from sqlalchemy.orm import Session
from app.crud.base import CRUDBase
from app.core.config import get_originated_from, OriginationEnum
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.ppg_common.schemas.bertopic.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate

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

    def get_latest_label_dictionary(self, db: Session, *, originated_from = OriginationEnum.ORIGINATED_FROM_APP) -> Union[list, None]:
        label_dictionary = None

        db_obj = db.query(self.model).filter(self.model.uploaded,
            self.model.originated_from == originated_from).order_by(self.model.version.desc()).first()

        if db_obj and 'labeling_terms' in db_obj.reference.keys():
            label_dictionary = db_obj.reference

        return label_dictionary

bertopic_embedding_pretrained = CRUDBertopicEmbeddingPretrained(
    BertopicEmbeddingPretrainedModel)
