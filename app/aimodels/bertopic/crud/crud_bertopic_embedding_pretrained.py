
from typing import Union
from sqlalchemy.orm import Session
from app.crud.base import CRUDBase
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from ..schemas.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])


class CRUDBertopicEmbeddingPretrained(CRUDBase[BertopicEmbeddingPretrainedModel, BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate]):
    def get_by_sha256(self, db: Session, *, sha256: str) -> Union[BertopicEmbeddingPretrainedModel, None]:
        if not sha256:
            return None

        return db.query(self.model).filter(self.model.sha256 == sha256).first()


bertopic_embedding_pretrained = CRUDBertopicEmbeddingPretrained(
    BertopicEmbeddingPretrainedModel)
