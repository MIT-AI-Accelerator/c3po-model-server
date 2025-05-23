import re
import enum
from typing import Optional
from pydantic import BaseModel, UUID4, field_validator, ConfigDict
from app.core.config import OriginationEnum

class EmbeddingModelTypeEnum(str, enum.Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    WEAK_LEARNERS = "weak_learners"
    DIFF_CSE = "diff_cse"
    CROSS_ENCODERS = "cross_encoders"

# Shared properties
class BertopicEmbeddingPretrainedBase(BaseModel):
    sha256: Optional[str] = None
    model_type: Optional[EmbeddingModelTypeEnum] = None
    model_name: Optional[str] = ''
    reference: Optional[dict] = {}

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    # ensure valid sha256 format
    @field_validator('sha256')
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
    reference: dict = {}

# Properties to receive on BertopicEmbeddingPretrained update
class BertopicEmbeddingPretrainedUpdate(BertopicEmbeddingPretrainedBase):
    uploaded: bool = False

# Properties shared by models stored in DB
class BertopicEmbeddingPretrainedInDBBase(BertopicEmbeddingPretrainedBase):
    id: UUID4
    model_type: EmbeddingModelTypeEnum
    model_name: str = ''
    uploaded: bool = False
    reference: dict = {}
    version: int
    sha256: str
    originated_from: OriginationEnum

# Properties to return to client
class BertopicEmbeddingPretrained(BertopicEmbeddingPretrainedInDBBase):
    pass

# Properties properties stored in DB
class BertopicEmbeddingPretrainedInDB(BertopicEmbeddingPretrainedInDBBase):
    pass
