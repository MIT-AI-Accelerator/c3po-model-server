import uuid
from typing import TYPE_CHECKING
from sqlalchemy import Column, UUID, String, ForeignKey, Enum
from sqlalchemy.orm import relationship
from ppg.config import OriginationEnum
from app.db.base_class import Base
from app.core.config import get_originated_from

if TYPE_CHECKING:
    from .mattermost_channels import MattermostChannelModel
    from .mattermost_users import MattermostUserModel
    from app.aimodels.bertopic.models.document import DocumentModel


class MattermostDocumentModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    message_id = Column(String(), unique=True)
    root_message_id = Column(String())
    channel = Column(UUID, ForeignKey("mattermostchannelmodel.id"))
    user = Column(UUID, ForeignKey("mattermostusermodel.id"))
    document = Column(UUID, ForeignKey("documentmodel.id"))
    originated_from = Column(Enum(OriginationEnum),
                             default=get_originated_from)
