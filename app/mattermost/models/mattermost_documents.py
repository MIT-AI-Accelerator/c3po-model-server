import uuid
from typing import TYPE_CHECKING
from sqlalchemy import Column, UUID, String, ForeignKey, Enum, JSON, Boolean, UniqueConstraint
from sqlalchemy.ext.mutable import MutableDict
from ppg.core.config import OriginationEnum
from ppg.schemas.mattermost.mattermost_documents import InfoTypeEnum, ThreadTypeEnum
from app.db.base_class import Base
from app.core.config import get_originated_from

if TYPE_CHECKING:
    from .mattermost_channels import MattermostChannelModel
    from .mattermost_users import MattermostUserModel
    from app.aimodels.bertopic.models.document import DocumentModel


class MattermostDocumentModel(Base):

    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    message_id = Column(String())
    root_message_id = Column(String())
    channel = Column(UUID, ForeignKey("mattermostchannelmodel.id", ondelete="CASCADE", onupdate="CASCADE"))
    user = Column(UUID, ForeignKey("mattermostusermodel.id", ondelete="CASCADE", onupdate="CASCADE"))
    document = Column(UUID, ForeignKey("documentmodel.id", ondelete="CASCADE", onupdate="CASCADE"))
    type = Column(String())
    hashtags = Column(String())
    has_reactions = Column(Boolean(), default=False)
    props = Column(MutableDict.as_mutable(JSON))
    doc_metadata = Column(MutableDict.as_mutable(JSON))
    thread_type = Column(Enum(ThreadTypeEnum), default=ThreadTypeEnum.MESSAGE)
    info_type = Column(Enum(InfoTypeEnum), default=InfoTypeEnum.CHAT)
    originated_from = Column(Enum(OriginationEnum),
                             default=get_originated_from)

    # mattermost message IDs must be unique,
    # allow for a single conversation thread for each message
    __table_args__ = (UniqueConstraint('message_id', 'thread_type', name='_messageid_threadtype_uc'),)
