from datetime import datetime
from pydantic import BaseModel, UUID4, ConfigDict
from typing import Optional
from app.core.config import OriginationEnum

# Shared properties
class DocumentBase(BaseModel):
    text: str
    original_created_time: Optional[datetime] = datetime.now()

# Properties to receive on Document creation
class DocumentCreate(DocumentBase):
    pass

# Properties to receive on Document update
class DocumentUpdate(DocumentBase):
    pass

# Properties shared by models stored in DB
class DocumentInDBBase(DocumentBase):
    id: UUID4
    originated_from: OriginationEnum

    model_config = ConfigDict(from_attributes=True)

# Properties to return to client
class Document(DocumentInDBBase):
    pass

# Properties properties stored in DB
class DocumentInDB(DocumentInDBBase):
    pass
