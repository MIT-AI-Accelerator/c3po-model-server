from typing import TYPE_CHECKING
from sqlalchemy import Column, UUID, String, Enum, JSON, Integer
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base
from app.core.config import OriginationEnum, get_originated_from
import uuid

class TopicSummaryModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
    model_id = Column(UUID)
    topic_id = Column(Integer)
    name = Column(String)
    top_n_words = Column(String, default="")
    top_n_documents = Column(MutableDict.as_mutable(JSON))
    summary = Column(String)
    trained_model = relationship("BertopicTrainedModel", back_populates="topic_summaries")
