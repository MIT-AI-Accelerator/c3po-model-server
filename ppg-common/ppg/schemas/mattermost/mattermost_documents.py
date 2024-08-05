import enum
from pydantic import BaseModel, UUID4
from ppg.core.config import OriginationEnum


class InfoTypeEnum(str, enum.Enum):
    CHAT = "chat"
    BOT = "bot"
    NOTAM = "notam"
    DATAMINR = "dataminr"
    ACARS = "acars"
    ENVISION = "envision"
    CAMPS = "camps"
    ARINC = "arinc"
    UDL = "udl"


# Shared properties
class MattermostDocumentBase(BaseModel):
    message_id: str
    root_message_id: str
    channel: UUID4
    user: UUID4
    document: UUID4
    type: str
    hashtags: str
    has_reactions: bool
    props: dict
    doc_metadata: dict
    is_thread: bool
    info_type: InfoTypeEnum

class MattermostDocumentCreate(MattermostDocumentBase):
    # Properties to receive on MattermostDocument creation

    pass

class MattermostDocumentUpdate(MattermostDocumentBase):
    # Properties to receive on MattermostDocument update

    pass

class MattermostDocumentInDBBase(MattermostDocumentBase):
    # Properties shared by models stored in DB

    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True


class MattermostDocument(MattermostDocumentInDBBase):
    # Properties to return to client

    pass


class MattermostDocumentInDB(MattermostDocumentInDBBase):
    # Properties properties stored in DB

    pass
