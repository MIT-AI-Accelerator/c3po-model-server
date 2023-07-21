
from typing import Union
from sqlalchemy.orm import Session
from app.crud.base import CRUDBase
from ..models.mattermost_channels import MattermostChannelModel
from ..schemas.mattermost_channels import MattermostChannelCreate
from ..models.mattermost_users import MattermostUserModel
from ..schemas.mattermost_users import MattermostUserCreate
from ..models.mattermost_documents import MattermostDocumentModel
from ..schemas.mattermost_documents import MattermostDocumentCreate

class CRUDMattermostChannel(CRUDBase[MattermostChannelModel, MattermostChannelCreate, MattermostChannelCreate]):
    def get_by_channel_id(self, db: Session, *, channel_id: str) -> Union[MattermostChannelModel, None]:
        if not channel_id:
            return None

        return db.query(self.model).filter(self.model.channel_id == channel_id).first()

    # note: use at your own risk until we are sure the team / channel name combinations are unique
    def get_by_channel_name(self, db: Session, *, team_name: str, channel_name: str) -> Union[MattermostChannelModel, None]:
        if not team_name or not channel_name:
            return None

        return db.query(self.model).filter(self.model.team_name == team_name, self.model.channel_name == channel_name).first()

    def get_all_channel_ids(self, db: Session) -> list[str]:
        channel_objs = db.query(self.model).all()
        return [obj.channel_id for obj in channel_objs]

class CRUDMattermostUser(CRUDBase[MattermostUserModel, MattermostUserCreate, MattermostUserCreate]):
    def get_by_user_id(self, db: Session, *, user_id: str) -> Union[MattermostUserModel, None]:
        if not user_id:
            return None

        return db.query(self.model).filter(self.model.user_id == user_id).first()

    def get_by_user_name(self, db: Session, *, user_name: str) -> Union[MattermostUserModel, None]:
        if not user_name:
            return None

        return db.query(self.model).filter(self.model.user_name == user_name).first()

class CRUDMattermostDocument(CRUDBase[MattermostDocumentModel, MattermostDocumentCreate, MattermostDocumentCreate]):
    def get_by_message_id(self, db: Session, *, message_id: str) -> Union[MattermostDocumentModel, None]:
        if not message_id:
            return None

        return db.query(self.model).filter(self.model.message_id == message_id).first()
        
    def get_all_channel_documents(self, db: Session, *, channels: list[str]) -> Union[list[MattermostDocumentModel], None]:
        if not channels:
            return None

        return sum([db.query(self.model).filter(self.model.channel == cuuid).all() for cuuid in channels], [])


mattermost_channels = CRUDMattermostChannel(MattermostChannelModel)
mattermost_users = CRUDMattermostUser(MattermostUserModel)
mattermost_documents = CRUDMattermostDocument(MattermostDocumentModel)
