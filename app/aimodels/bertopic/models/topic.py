from typing import TYPE_CHECKING
from sqlalchemy import Column, UUID, String, Enum, JSON, Integer, Boolean, ForeignKey
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ppg.core.config import OriginationEnum
from app.db.base_class import Base
from app.core.config import get_originated_from
import uuid

class TopicSummaryModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
    model_id = Column(UUID, ForeignKey("bertopictrainedmodel.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=True)
    topic_id = Column(Integer)
    name = Column(String)
    top_n_words = Column(String, default="")
    top_n_documents = Column(MutableDict.as_mutable(JSON))
    summary = Column(String)
    is_trending = Column(Boolean(), default=False)
