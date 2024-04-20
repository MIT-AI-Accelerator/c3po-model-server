import re
import enum
from typing import Optional
from pydantic import BaseModel, UUID4, validator
from ppg.core.config import OriginationEnum

class LlmFilenameEnum(str, enum.Enum):
    L13B_SNOOZY = "ggml-gpt4all-l13b-snoozy.bin"
    Q4_K_M = "mistrallite.Q4_K_M.gguf"

# Shared properties
class LlmPretrainedBase(BaseModel):
    sha256: Optional[str] = None
    model_type: Optional[LlmFilenameEnum] = None
    use_base_model: Optional[bool] = None

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

# Properties to receive on LlmPretrained creation
class LlmPretrainedCreate(LlmPretrainedBase):
    sha256: str
    use_base_model: bool = False

# Properties to receive on LlmPretrained update
class LlmPretrainedUpdate(LlmPretrainedBase):
    uploaded: bool

# Properties shared by models stored in DB
class LlmPretrainedInDBBase(LlmPretrainedBase):
    id: UUID4
    model_type: LlmFilenameEnum
    uploaded: bool
    version: int
    sha256: str
    use_base_model: bool
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class LlmPretrained(LlmPretrainedInDBBase):
    pass

# Properties properties stored in DB
class LlmPretrainedInDB(LlmPretrainedInDBBase):
    pass
