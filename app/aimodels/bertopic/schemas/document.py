from datetime import datetime
from pydantic import BaseModel, UUID4
from app.core.config import OriginationEnum

# Shared properties
class DocumentBase(BaseModel):
    text: str

# Properties to receive on Document creation
class DocumentCreate(DocumentBase):
    pass

# Properties shared by models stored in DB
class DocumentInDBBase(DocumentBase):
    id: UUID4
    original_created_time: datetime
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class Document(DocumentInDBBase):
    pass

# Properties properties stored in DB
class DocumentInDB(DocumentInDBBase):
    pass
