from typing import Union
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session
from ppg.schemas.mattermost.mattermost_channels import MattermostChannelCreate
from ppg.schemas.mattermost.mattermost_documents import MattermostDocumentCreate
from ppg.schemas.mattermost.mattermost_users import MattermostUserCreate, MattermostUserUpdate
import ppg.services.mattermost_utils as mattermost_utils
from app.core.config import settings
from app.core.logging import logger
from app.crud.base import CRUDBase
from ..models.mattermost_channels import MattermostChannelModel
from ..models.mattermost_users import MattermostUserModel
from ..models.mattermost_documents import MattermostDocumentModel
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

    def get_all_channel_documents(self, db: Session, channels: list[str], history_depth: int = 0) -> Union[list[MattermostDocumentModel], None]:

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
            documents = sum([db.query(self.model).filter(
                self.model.channel == cuuid).all() for cuuid in channels], [])

        return documents

    def get_document_dataframe(self, db: Session, *, mm_document_uuids: list[str]) -> Union[pd.DataFrame, None]:

        ddf = pd.DataFrame()
        for duuid in mm_document_uuids:
            document = db.query(self.model, DocumentModel, MattermostUserModel, MattermostChannelModel).join(DocumentModel, DocumentModel.id == self.model.document).join(
                MattermostUserModel, MattermostUserModel.id == self.model.user).join(MattermostChannelModel, MattermostChannelModel.id == self.model.channel).filter(self.model.id == duuid).all()
            if document:
                ddf = pd.concat([ddf, pd.DataFrame([{'id': document[0][0].message_id,
                                                     'message': document[0][1].text,
                                                     'root_id': document[0][0].root_message_id,
                                                     'type': document[0][0].type,
                                                     'user_id': document[0][2].user_id,
                                                     'channel_id': document[0][3].channel_id,
                                                     'create_at': document[0][1].original_created_time}])])

        return ddf
    
    def get_document_dataframe_by_documents(self, db: Session, *, document_uuids: list[str]) -> Union[pd.DataFrame, None]:

        ddf = pd.DataFrame()
        for duuid in document_uuids:
            mm_document = db.query(self.model, DocumentModel, MattermostUserModel, MattermostChannelModel).join(DocumentModel, DocumentModel.id == self.model.document).join(
                MattermostUserModel, MattermostUserModel.id == self.model.user).join(MattermostChannelModel, MattermostChannelModel.id == self.model.channel).filter(DocumentModel.id == duuid).all()
            if mm_document:
                ddf = pd.concat([ddf, pd.DataFrame([{'uuid': duuid,
                                                    'id': mm_document[0][0].message_id,
                                                    'message': mm_document[0][1].text,
                                                    'root_id': mm_document[0][0].root_message_id,
                                                    'type': mm_document[0][0].type,
                                                    'user_id': mm_document[0][2].user_id,
                                                    'channel_id': mm_document[0][3].channel_id,
                                                    'create_at': mm_document[0][1].original_created_time}])])
            else:
                document = db.query(DocumentModel).filter(DocumentModel.id == duuid).first()
                ddf = pd.concat([ddf, pd.DataFrame([{'uuid': duuid,
                                                    'id': None,
                                                    'message': document.text,
                                                    'root_id': None,
                                                    'type': None,
                                                    'user_id': None,
                                                    'channel_id': None,
                                                    'create_at': document.original_created_time}])])
                
        return ddf


def populate_mm_user_info(db: Session, *, mm_user: dict, teams: dict) -> MattermostUserModel:

    user_obj = mattermost_users.get_by_user_name(db, user_name=mm_user['username'])
    if user_obj is None:
        create_user = MattermostUserCreate(
            user_id=mm_user['id'],
            user_name=mm_user['username'],
            nickname=mm_user['nickname'],
            first_name=mm_user['first_name'],
            last_name=mm_user['last_name'],
            position=mm_user['position'],
            email=mm_user['email'],
            teams=teams)
        user_obj = mattermost_users.create(db, obj_in=create_user)
    else:
        update_user = MattermostUserUpdate(
            user_id=mm_user['id'],
            user_name=mm_user['username'],
            nickname=mm_user['nickname'],
            first_name=mm_user['first_name'],
            last_name=mm_user['last_name'],
            position=mm_user['position'],
            email=mm_user['email'],
            teams=teams)
        user_obj = mattermost_users.update(
            db, db_obj=user_obj, obj_in=update_user)

    if user_obj is None:
        logger.debug(f"Unable to retrieve user: {user_name}")

    return user_obj


def populate_mm_user_team_info(db: Session, *, user_name: str, get_teams = False) -> MattermostUserModel:

    # add or update user info in db
    (mm_user, tdf) = mattermost_utils.get_user_info(
        settings.mm_base_url, settings.mm_token, user_name, get_teams)
    if mm_user is None:
        logger.debug(f"Mattermost user not found: {user_name}")
        return None

    teams = dict()
    if not tdf.empty:
        teams = tdf['name'].to_dict()
    elif get_teams:
        logger.debug(f"Unable to access teams for user: {user_name}")

    user_obj = populate_mm_user_info(db, mm_user=mm_user, teams=teams)

    # retrieve existing channels from db
    cdf = mattermost_utils.get_all_user_channels(
        settings.mm_base_url, settings.mm_token, mm_user['id'], teams)
    if not cdf.empty:
        existing_ids = mattermost_channels.get_all_channel_ids(db)
        cdf = cdf[~cdf.id.isin(existing_ids)]

        # group messages including the user will appear as duplicate channels
        # We think 'O' for a public channel, 'P' for a private channel, 'D' for direct message, 'G' for group message
        # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=44028044
        v = cdf.id.value_counts()
        dcdf = cdf[cdf.id.isin(v.index[v.gt(1)])]
        dcdf = dcdf[~dcdf['type'].str.contains("G")]
        dcdf = dcdf[~dcdf['type'].str.contains("D")]
        if not dcdf.empty:
            logger.warn(f"Duplicate Mattermost channels found: {dcdf.id.unique()}")
        cdf.drop_duplicates(subset=['id'], keep='first', inplace=True)

        # add new channels to db
        channels = [MattermostChannelCreate(
            channel_id=row['id'],
            channel_name=row['name'],
            team_id=row['team_id'],
            team_name=row['team_name'],
            display_name=row['display_name'],
            type=row['type'],
            header=row['header'],
            purpose=row['purpose']
        ) for key, row in cdf.iterrows()]
        new_channels = mattermost_channels.create_all_using_id(
            db, obj_in_list=channels)

        if new_channels is None:
            logger.debug(
                f"Unable to create user channels: {[c.channel_id for c in channels]}")

    return user_obj


def populate_mm_channel_info(db: Session, *, channel_info: dict) -> MattermostChannelModel:
    channel_obj = mattermost_channels.get_by_channel_id(db, channel_id=channel_info['id'])
    if not channel_obj:
        channel = MattermostChannelCreate(
            channel_id=channel_info['id'],
            channel_name=channel_info['name'],
            team_id=channel_info['team_id'],
            team_name=channel_info['team_name'],
            display_name=channel_info['display_name'],
            type=channel_info['type'],
            header=channel_info['header'],
            purpose=channel_info['purpose'])
        channel_obj = mattermost_channels.create(db, obj_in=channel)
    else:
        logger.warn(f"Duplicate Mattermost channel found: {channel_info['id']}")
    return channel_obj


# Takes message utterances (i.e. individual rows) from chat dataframe, and converts them to conversation threads
# Returns a dataframe with original structure; messages updated to include full conversation
def convert_conversation_threads(df: pd.DataFrame):

    df['root_id'] = df['root_id'].fillna('')
    df['message'] = df['message'].fillna('')
    threads = {}
    threads_row = {}
    for index, row in df.iterrows():
        thread = row['root_id']
        utterance = row['message']
        p_id = row['id']
        if utterance.find("added to the channel") < 0 and utterance.find("joined the channel") < 0 and utterance.find("left the channel") < 0:
            if len(thread) > 0:
                if thread not in threads:
                    threads[thread] = [utterance.replace("\n", " ")]
                else:
                    threads[thread].append(utterance.replace("\n", " "))
            else:
                t = []
                t.append(utterance.replace("\n", " "))
                threads[p_id] = t
                threads_row[p_id] = row
    keys = set(sorted(threads.keys())).intersection(threads_row.keys())

    data = []
    for index, key in enumerate(keys):
        row = threads_row[key]
        row['message'] = "\n".join(threads[key])
        data.append(row)

    return pd.DataFrame(data, columns=df.columns)


mattermost_channels = CRUDMattermostChannel(MattermostChannelModel)
mattermost_users = CRUDMattermostUser(MattermostUserModel)
mattermost_documents = CRUDMattermostDocument(MattermostDocumentModel)
