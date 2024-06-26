import uuid
from typing import TYPE_CHECKING
from sqlalchemy import Boolean, Column, ForeignKey, UUID, Enum, String, DateTime, JSON
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ppg.core.config import OriginationEnum
from app.db.base_class import Base
from app.core.config import get_originated_from

if TYPE_CHECKING:
    from .document import DocumentModel  # noqa: F401
    from .bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel  # noqa: F401

# JSON won't be mutable, see here if that needs to change: https://stackoverflow.com/questions/1378325/python-dicts-in-sqlalchemy
class BertopicTrainedModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    time = Column(DateTime(timezone=True), server_default=func.now())
    uploaded = Column(Boolean(), default=False)
    sentence_transformer_id = Column(UUID)
    weak_learner_id = Column(UUID)
    summarization_model_id = Column(UUID)
    seed_topics = Column(MutableDict.as_mutable(JSON))
    stop_words = Column(MutableDict.as_mutable(JSON))
    prompt_template = Column(String)
    refine_template = Column(String)
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)

    embedding_pretrained_id = Column(UUID, ForeignKey("bertopicembeddingpretrainedmodel.id", ondelete="CASCADE", onupdate="CASCADE"))
    embedding_pretrained = relationship("BertopicEmbeddingPretrainedModel", back_populates="trained_models")

    trained_on_documents = relationship("DocumentModel", secondary="documentbertopictrainedmodel", back_populates="used_in_trained_models")
