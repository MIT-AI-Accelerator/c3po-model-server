from typing import Optional

from pydantic import BaseModel, UUID4
from ..models.bertopic_embedding_pretrained import EmbeddingModelTypeEnum

# Shared properties
class BertopicEmbeddingPretrainedBase(BaseModel):
    sha256: Optional[str] = None
    model_type: Optional[EmbeddingModelTypeEnum] = None

# Properties to receive on BertopicEmbeddingPretrained creation
class BertopicEmbeddingPretrainedCreate(BertopicEmbeddingPretrainedBase):
    sha256: str

# Properties to receive on BertopicEmbeddingPretrained update
class BertopicEmbeddingPretrainedUpdate(BertopicEmbeddingPretrainedBase):
    uploaded: bool = False

# Properties shared by models stored in DB
class BertopicEmbeddingPretrainedInDBBase(BertopicEmbeddingPretrainedBase):
    id: UUID4
    model_type: EmbeddingModelTypeEnum
    uploaded: bool = False
    version: int
    sha256: str

    class Config:
        orm_mode = True

# Properties to return to client
class BertopicEmbeddingPretrained(BertopicEmbeddingPretrainedInDBBase):
    pass

# Properties properties stored in DB
class BertopicEmbeddingPretrainedInDB(BertopicEmbeddingPretrainedInDBBase):
    pass
