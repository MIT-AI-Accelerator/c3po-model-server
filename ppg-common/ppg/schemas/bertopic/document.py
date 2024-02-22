from datetime import datetime
from pydantic import BaseModel, UUID4
from typing import Optional
from ppg.core.config import OriginationEnum

# Shared properties
class DocumentBase(BaseModel):
    text: str
    original_created_time: Optional[datetime] = datetime.now()

# Properties to receive on Document creation
class DocumentCreate(DocumentBase):
    pass

# Properties shared by models stored in DB
class DocumentInDBBase(DocumentBase):
    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class Document(DocumentInDBBase):
    pass

# Properties properties stored in DB
class DocumentInDB(DocumentInDBBase):
    pass
