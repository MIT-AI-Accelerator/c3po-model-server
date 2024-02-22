import uuid
from typing import TYPE_CHECKING
from sqlalchemy import Column, UUID, String, Enum, JSON
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from ppg.core.config import OriginationEnum
from app.db.base_class import Base
from app.core.config import get_originated_from

if TYPE_CHECKING:
    from .mattermost_documents import MattermostDocumentModel


class MattermostUserModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    user_id = Column(String(), unique=True)
    user_name = Column(String(), unique=True)
    nickname = Column(String())
    first_name = Column(String())
    last_name = Column(String())
    position = Column(String())
    email = Column(String())
    teams = Column(MutableDict.as_mutable(JSON))
    originated_from = Column(Enum(OriginationEnum),
                             default=get_originated_from)
    document_user = relationship(
        "MattermostDocumentModel", primaryjoin="MattermostUserModel.id==MattermostDocumentModel.user")
