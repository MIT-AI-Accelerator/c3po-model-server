from pydantic import BaseModel, UUID4
from ppg.core.config import OriginationEnum


class MattermostChannelBase(BaseModel):
    # Shared properties

    channel_id: str
    channel_name: str
    team_id: str
    team_name: str
    display_name: str
    type: str
    header: str
    purpose: str


class MattermostChannelCreate(MattermostChannelBase):
    # Properties to receive on MattermostChannel creation

    pass


class MattermostChannelInDBBase(MattermostChannelBase):
    # Properties shared by models stored in DB

    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True


class MattermostChannel(MattermostChannelInDBBase):
    # Properties to return to client

    pass


class MattermostChannelInDB(MattermostChannelInDBBase):
    # Properties properties stored in DB

    pass
