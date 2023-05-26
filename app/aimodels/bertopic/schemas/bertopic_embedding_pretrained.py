import re
from typing import Optional

from pydantic import BaseModel, UUID4, validator
from ..models.bertopic_embedding_pretrained import EmbeddingModelTypeEnum

# Shared properties
class BertopicEmbeddingPretrainedBase(BaseModel):
    sha256: Optional[str] = None
    model_type: Optional[EmbeddingModelTypeEnum] = None
    model_name: Optional[str] = None

    # ensure valid sha256 format
    @validator('sha256')
    def sha256_must_be_valid(cls, v):
        # pylint: disable=no-self-argument

        if v is None:
            return v

        lower_v = v.lower()
        if not re.match(r'^[a-f0-9]{64}$', lower_v):
            raise ValueError(
                'sha256 must be hexademical and 64 characters long')

        return lower_v

# Properties to receive on BertopicEmbeddingPretrained creation
class BertopicEmbeddingPretrainedCreate(BertopicEmbeddingPretrainedBase):
    sha256: str
    model_name: str = ''

# Properties to receive on BertopicEmbeddingPretrained update
class BertopicEmbeddingPretrainedUpdate(BertopicEmbeddingPretrainedBase):
    uploaded: bool = False

# Properties shared by models stored in DB
class BertopicEmbeddingPretrainedInDBBase(BertopicEmbeddingPretrainedBase):
    id: UUID4
    model_type: EmbeddingModelTypeEnum
    model_name: str = ''
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
