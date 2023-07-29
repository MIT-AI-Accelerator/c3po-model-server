from typing import TYPE_CHECKING
from sqlalchemy import Column, DateTime, UUID, String, Enum, ARRAY, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base
from app.core.config import OriginationEnum, get_originated_from
import uuid

# class TopicDocument(Base): TODO
#     text: str
#     relation: float

class TopicSummaryModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
    model_id = Column(UUID)
    topic_id = Column(Integer)
    name = Column(String)
    top_n_words = Column(String, default="")
    # documents = Column(ARRAY(TopicDocument, dimensions=1)) TODO
    summary = Column(String)
    
    # trained_model = relationship("BertopicTrainedModel", back_populates="Topic") TODO
