from typing import TYPE_CHECKING
from sqlalchemy import Boolean, Column, ForeignKey, UUID, Enum, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base
from app.core.config import OriginationEnum, get_originated_from
import uuid

if TYPE_CHECKING:
    from .document import DocumentModel  # noqa: F401
    from .bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel  # noqa: F401

# JSON won't be mutable, see here if that needs to change: https://stackoverflow.com/questions/1378325/python-dicts-in-sqlalchemy
class BertopicTrainedModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    time = Column(DateTime(timezone=True), server_default=func.now())
    uploaded = Column(Boolean(), default=False)
    sentence_transformer_id = Column(UUID)
    weak_leaner_id = Column(UUID)
    summarization_model_id = Column(UUID)
    # TODO seed_topics
    topic_word_visualization = Column(String)
    topic_cluster_visualization = Column(String)
    embedding_pretrained_id = Column(UUID, ForeignKey("bertopicembeddingpretrainedmodel.id"))
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
    embedding_pretrained = relationship("BertopicEmbeddingPretrainedModel", back_populates="trained_models")
    trained_on_documents = relationship("DocumentModel", secondary="documentbertopictrainedmodel", back_populates="used_in_trained_models")
