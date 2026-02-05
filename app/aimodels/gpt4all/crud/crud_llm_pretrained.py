
from typing import Union
from sqlalchemy.orm import Session
from app.core.config import OriginationEnum
from app.crud.base import CRUDBase
from app.ppg_common.schemas.gpt4all.llm_pretrained import LlmPretrainedCreate, LlmPretrainedUpdate
from ..models import LlmPretrainedModel
from ..models.llm_pretrained import LlmFilenameEnum

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDLlmPretrained(CRUDBase[LlmPretrainedModel, LlmPretrainedCreate, LlmPretrainedUpdate]):
    def get_by_sha256(self, db: Session, *, sha256: str) -> Union[LlmPretrainedModel, None]:
        if not sha256:
            return None

        return db.query(self.model).filter(self.model.sha256 == sha256).first()

    def get_latest_uploaded_by_model_type(self, db: Session, *, model_type: LlmFilenameEnum, originated_from = OriginationEnum.ORIGINATED_FROM_APP) -> Union[LlmPretrainedModel, None]:
        if not isinstance(model_type, LlmFilenameEnum):
            return None

        return db.query(self.model).filter(self.model.model_type == model_type,
                                           self.model.uploaded == True,
                                           self.model.originated_from == originated_from).order_by(self.model.version.desc()).first()


llm_pretrained = CRUDLlmPretrained(
    LlmPretrainedModel)
