from app.core.config import OriginationEnum

from pydantic import BaseModel, UUID4

# Shared properties
class MattermostDocumentBase(BaseModel):
    message_id: str
    channel: UUID4
    user: UUID4
    document: UUID4

# Properties to receive on MattermostDocument creation
class MattermostDocumentCreate(MattermostDocumentBase):
    pass

# Properties shared by models stored in DB
class MattermostDocumentInDBBase(MattermostDocumentBase):
    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class MattermostDocument(MattermostDocumentInDBBase):
    pass

# Properties properties stored in DB
class MattermostDocumentInDB(MattermostDocumentInDBBase):
    pass
