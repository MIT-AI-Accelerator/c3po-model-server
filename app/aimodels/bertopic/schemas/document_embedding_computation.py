from typing import Optional
from app.core.config import OriginationEnum

from pydantic import BaseModel, UUID4

# Shared properties
class DocumentEmbeddingComputationBase(BaseModel):
    embedding_vector: list
    document_id: Optional[UUID4] = None
    bertopic_embedding_pretrained_id: Optional[UUID4] = None

# Properties to receive on DocumentEmbeddingComputation creation
class DocumentEmbeddingComputationCreate(DocumentEmbeddingComputationBase):
    document_id: UUID4
    bertopic_embedding_pretrained_id: UUID4

# Properties to receive on DocumentEmbeddingComputation update
class DocumentEmbeddingComputationUpdate(DocumentEmbeddingComputationBase):
    pass

# Properties shared by models stored in DB
class DocumentEmbeddingComputationInDBBase(DocumentEmbeddingComputationBase):
    id: UUID4
    document_id: UUID4
    bertopic_embedding_pretrained_id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class DocumentEmbeddingComputation(DocumentEmbeddingComputationInDBBase):
    pass

# Properties properties stored in DB
class DocumentEmbeddingComputationInDB(DocumentEmbeddingComputationInDBBase):
    pass
