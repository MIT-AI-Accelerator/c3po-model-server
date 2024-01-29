from app.crud.base import CRUDBase
from ..models.document_embedding_computation import DocumentEmbeddingComputationModel
from ..schemas.document_embedding_computation import DocumentEmbeddingComputationCreate, DocumentEmbeddingComputationUpdate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDDocumentEmbeddingComputation(CRUDBase[DocumentEmbeddingComputationModel, DocumentEmbeddingComputationCreate, DocumentEmbeddingComputationUpdate]  ):
    pass

document_embedding_computation = CRUDDocumentEmbeddingComputation(DocumentEmbeddingComputationModel)
