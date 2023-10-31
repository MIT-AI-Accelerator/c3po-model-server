
from typing import Union
from sqlalchemy.orm import Session
from app.core.config import OriginationEnum, get_originated_from
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

    def get_latest_uploaded_by_model_type(self, db: Session, *, model_type: Gpt4AllModelFilenameEnum, originated_from = OriginationEnum.ORIGINATED_FROM_APP) -> Union[Gpt4AllPretrainedModel, None]:
        if not isinstance(model_type, Gpt4AllModelFilenameEnum):
            return None

        return db.query(self.model).filter(self.model.model_type == model_type,
                                           self.model.uploaded == True,
                                           self.model.originated_from == originated_from).order_by(self.model.version.desc()).first()


gpt4all_pretrained = CRUDGpt4AllPretrained(
    Gpt4AllPretrainedModel)
