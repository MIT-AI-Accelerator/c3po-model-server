from pydantic import BaseModel, UUID4
from app.core.config import OriginationEnum


class MattermostUserBase(BaseModel):
    # Shared properties

    user_id: str
    user_name: str
    teams: dict


class MattermostUserCreate(MattermostUserBase):
    # Properties to receive on MattermostUser creation

    pass


class MattermostUserUpdate(MattermostUserBase):
    # Properties to receive on MattermostUser creation

    teams: dict


class MattermostUserInDBBase(MattermostUserBase):
    # Properties shared by models stored in DB

    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True


class MattermostUser(MattermostUserInDBBase):
    # Properties to return to client

    pass


class MattermostUserInDB(MattermostUserInDBBase):
    # Properties properties stored in DB

    pass
