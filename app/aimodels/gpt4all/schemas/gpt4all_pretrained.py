import re
from typing import Optional

from pydantic import BaseModel, UUID4, validator
from ..models.gpt4all_pretrained import Gpt4AllModelFilenameEnum

# Shared properties
class Gpt4AllPretrainedBase(BaseModel):
    sha256: Optional[str] = None
    model_type: Optional[Gpt4AllModelFilenameEnum] = None
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

# Properties to receive on Gpt4AllPretrained creation
class Gpt4AllPretrainedCreate(Gpt4AllPretrainedBase):
    sha256: str
    use_base_model: bool = False

# Properties to receive on Gpt4AllPretrained update
class Gpt4AllPretrainedUpdate(Gpt4AllPretrainedBase):
    uploaded: bool

# Properties shared by models stored in DB
class Gpt4AllPretrainedInDBBase(Gpt4AllPretrainedBase):
    id: UUID4
    model_type: Gpt4AllModelFilenameEnum
    uploaded: bool
    version: int
    sha256: str
    use_base_model: bool

    class Config:
        orm_mode = True

# Properties to return to client
class Gpt4AllPretrained(Gpt4AllPretrainedInDBBase):
    pass

# Properties properties stored in DB
class Gpt4AllPretrainedInDB(Gpt4AllPretrainedInDBBase):
    pass
