from typing import Union
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.crud.base import CRUDBase
from ..services import mattermost_utils
from ..models.mattermost_channels import MattermostChannelModel
from ..schemas.mattermost_channels import MattermostChannelCreate
from ..models.mattermost_users import MattermostUserModel
from ..schemas.mattermost_users import MattermostUserCreate, MattermostUserUpdate
from ..models.mattermost_documents import MattermostDocumentModel
from ..schemas.mattermost_documents import MattermostDocumentCreate
from app.aimodels.bertopic.models import DocumentModel

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

    def get_all_channel_documents(self, db: Session, channels: list[str], history_depth: int=0) -> Union[list[MattermostDocumentModel], None]:

        # get documents <= history_depth days old
        if history_depth > 0:
            ctime = datetime.now()
            stime = ctime - pd.DateOffset(days=history_depth)
            documents = sum([db.query(self.model).join(DocumentModel).filter(self.model.channel == cuuid,
                                                                             DocumentModel.original_created_time >= stime,
                                                                             DocumentModel.original_created_time <= ctime)
                                                                             .all() for cuuid in channels], [])

        # get all documents
        else:
            documents = sum([db.query(self.model).filter(self.model.channel == cuuid).all() for cuuid in channels], [])

        return documents


def populate_mm_user_info(db: Session, *, user_name: str) -> MattermostUserModel:

    # add or update user info in db
    (mm_user, tdf) = mattermost_utils.get_user_info(
        settings.mm_base_url, settings.mm_token, user_name)
    if mm_user is None:
        logger.debug(f"Mattermost user not found: {user_name}")
        return None

    teams = dict()
    if not tdf.empty:
        teams = tdf['name'].to_dict()
    else:
        logger.debug(f"Unable to access teams for user: {user_name}")

    user_obj = mattermost_users.get_by_user_name(db, user_name=user_name)
    if user_obj is None:
        create_user = MattermostUserCreate(
            user_id=mm_user['id'], user_name=mm_user['username'], teams=teams)
        user_obj = mattermost_users.create(db, obj_in=create_user)
    else:
        update_user = MattermostUserUpdate(
            user_id=mm_user['id'], user_name=mm_user['username'], teams=teams)
        user_obj = mattermost_users.update(
            db, db_obj=user_obj, obj_in=update_user)

    if user_obj is None:
        logger.debug(f"Unable to retrieve user: {user_name}")

    # retrieve existing channels from db
    cdf = mattermost_utils.get_all_user_channels(
        settings.mm_base_url, settings.mm_token, mm_user['id'], teams)
    if not cdf.empty:
        existing_ids = mattermost_channels.get_all_channel_ids(db)
        cdf = cdf[~cdf.id.isin(existing_ids)]

        # add new channels to db
        channels = [MattermostChannelCreate(
            channel_id=row['id'],
            channel_name=row['name'],
            team_id=row['team_id'],
            team_name=row['team_name']
        ) for key, row in cdf.iterrows()]
        new_channels = mattermost_channels.create_all_using_id(
            db, obj_in_list=channels)

        if new_channels is None:
            logger.debug(
                f"Unable to create user channels: {[c.channel_id for c in channels]}")

    return user_obj


def populate_mm_channel_info(db: Session, *, channel_info: dict) -> MattermostChannelModel:
    channel = MattermostChannelCreate(
        channel_id=channel_info['id'],
        channel_name=channel_info['name'],
        team_id=channel_info['team_id'],
        team_name=channel_info['team_name'])
    return mattermost_channels.create(db, obj_in=channel)


mattermost_channels = CRUDMattermostChannel(MattermostChannelModel)
mattermost_users = CRUDMattermostUser(MattermostUserModel)
mattermost_documents = CRUDMattermostDocument(MattermostDocumentModel)
