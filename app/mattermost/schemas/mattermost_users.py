from app.core.config import OriginationEnum
from pydantic import BaseModel, UUID4

# Shared properties
class MattermostUserBase(BaseModel):
    user_id: str
    user_name: str
    teams: dict

# Properties to receive on MattermostUser creation
class MattermostUserCreate(MattermostUserBase):
    pass

# Properties to receive on MattermostUser creation
class MattermostUserUpdate(MattermostUserBase):
    teams: dict

# Properties shared by models stored in DB
class MattermostUserInDBBase(MattermostUserBase):
    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class MattermostUser(MattermostUserInDBBase):
    pass

# Properties properties stored in DB
class MattermostUserInDB(MattermostUserInDBBase):
    pass
