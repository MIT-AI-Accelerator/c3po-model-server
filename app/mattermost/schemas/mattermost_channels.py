from app.core.config import OriginationEnum

from pydantic import BaseModel, UUID4

# Shared properties
class MattermostChannelBase(BaseModel):
    channel_id: str
    channel_name: str
    team_id: str
    team_name: str

# Properties to receive on MattermostChannel creation
class MattermostChannelCreate(MattermostChannelBase):
    pass

# Properties shared by models stored in DB
class MattermostChannelInDBBase(MattermostChannelBase):
    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class MattermostChannel(MattermostChannelInDBBase):
    pass

# Properties properties stored in DB
class MattermostChannelInDB(MattermostChannelInDBBase):
    pass
