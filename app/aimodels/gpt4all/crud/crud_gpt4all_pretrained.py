
from typing import Union
from sqlalchemy.orm import Session
from app.crud.base import CRUDBase
from ..models import Gpt4AllPretrainedModel
from ..models.gpt4all_pretrained import Gpt4AllModelFilenameEnum
from ..schemas import Gpt4AllPretrainedCreate, Gpt4AllPretrainedUpdate
from sqlalchemy import func

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDGpt4AllPretrained(CRUDBase[Gpt4AllPretrainedModel, Gpt4AllPretrainedCreate, Gpt4AllPretrainedUpdate]):
    def get_by_sha256(self, db: Session, *, sha256: str) -> Union[Gpt4AllPretrainedModel, None]:
        if not sha256:
            return None

        return db.query(self.model).filter(self.model.sha256 == sha256).first()

    def get_latest_uploaded_by_model_type(self, db: Session, *, model_type: Gpt4AllModelFilenameEnum) -> Union[Gpt4AllPretrainedModel, None]:
        if not isinstance(model_type, Gpt4AllModelFilenameEnum):
            return None

        # see here for info: https://stackoverflow.com/questions/30784456/sqlalchemy-return-a-record-filtered-by-max-value-of-a-column
        subqry = db.query(func.max(self.model.version)).filter(self.model.model_type == model_type, self.model.uploaded == True)
        return db.query(self.model).filter(self.model.model_type == model_type, self.model.version == subqry).first()


gpt4all_pretrained = CRUDGpt4AllPretrained(
    Gpt4AllPretrainedModel)
