
from app.crud.base import CRUDBase
from ..models.document import DocumentModel
from ..schemas.document import DocumentCreate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDDocument(CRUDBase[DocumentModel, DocumentCreate, DocumentCreate]  ):
    pass

document = CRUDDocument(DocumentModel)
