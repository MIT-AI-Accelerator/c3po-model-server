import uuid
from typing import TYPE_CHECKING
from sqlalchemy import Column, UUID, String, Enum, UniqueConstraint
from sqlalchemy.orm import relationship
from app.db.base_class import Base
from app.core.config import get_originated_from, OriginationEnum

if TYPE_CHECKING:
    pass


class MattermostChannelModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    channel_id = Column(String(), unique=True)
    channel_name = Column(String())
    team_id = Column(String())
    team_name = Column(String())
    display_name = Column(String())
    type = Column(String())
    header = Column(String())
    purpose = Column(String())
    originated_from = Column(Enum(OriginationEnum),
                             default=get_originated_from)
    document_channel = relationship(
        "MattermostDocumentModel", primaryjoin="MattermostChannelModel.id==MattermostDocumentModel.channel")
    team_channel_pairs = UniqueConstraint(
        'team_name', 'channel_name', name='team_channel_pairs')
