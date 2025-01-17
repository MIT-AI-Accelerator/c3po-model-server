from app.crud.base import CRUDBase
from app.ppg_common.schemas.bertopic.document_embedding_computation import DocumentEmbeddingComputationCreate, DocumentEmbeddingComputationUpdate
from ..models.document_embedding_computation import DocumentEmbeddingComputationModel

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDDocumentEmbeddingComputation(CRUDBase[DocumentEmbeddingComputationModel, DocumentEmbeddingComputationCreate, DocumentEmbeddingComputationUpdate]  ):
    pass

document_embedding_computation = CRUDDocumentEmbeddingComputation(DocumentEmbeddingComputationModel)
