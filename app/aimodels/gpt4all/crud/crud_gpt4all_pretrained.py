
from typing import Union
from sqlalchemy.orm import Session
from app.crud.base import CRUDBase
from ..models import Gpt4AllPretrainedModel
from ..schemas import Gpt4AllPretrainedCreate, Gpt4AllPretrainedUpdate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDGpt4AllPretrained(CRUDBase[Gpt4AllPretrainedModel, Gpt4AllPretrainedCreate, Gpt4AllPretrainedUpdate]):
    def get_by_sha256(self, db: Session, *, sha256: str) -> Union[Gpt4AllPretrainedModel, None]:
        if not sha256:
            return None

        return db.query(self.model).filter(self.model.sha256 == sha256).first()


gpt4all_pretrained = CRUDGpt4AllPretrained(
    Gpt4AllPretrainedModel)
