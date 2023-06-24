from typing import TYPE_CHECKING
from sqlalchemy import Boolean, Column, ForeignKey, UUID, JSON, Enum
from sqlalchemy.orm import relationship
from app.db.base_class import Base
from app.core.config import OriginationEnum, get_originated_from
import uuid

if TYPE_CHECKING:
    from .document import DocumentModel  # noqa: F401
    from .bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel  # noqa: F401

# JSON won't be mutable, see here if that needs to change: https://stackoverflow.com/questions/1378325/python-dicts-in-sqlalchemy
class BertopicTrainedModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    uploaded = Column(Boolean(), default=False)
    plotly_bubble_config = Column(JSON)
    embedding_pretrained_id = Column(UUID, ForeignKey("bertopicembeddingpretrainedmodel.id"))
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
    embedding_pretrained = relationship("BertopicEmbeddingPretrainedModel", back_populates="trained_models")
    trained_on_documents = relationship("DocumentModel", secondary="documentbertopictrainedmodel", back_populates="used_in_trained_models")
