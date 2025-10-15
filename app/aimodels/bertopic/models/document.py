from typing import TYPE_CHECKING
from sqlalchemy import Column, DateTime, UUID, String, Enum
from sqlalchemy.orm import relationship
from app.db.base_class import Base
from app.core.config import get_originated_from, OriginationEnum
import uuid

if TYPE_CHECKING:
    from .document_embedding_computation import DocumentEmbeddingComputationModel  # noqa: F401
    from .bertopic_trained import BertopicTrainedModel  # noqa: F401
    from app.mattermost.models.mattermost_documents import MattermostDocumentModel  # noqa: F401

class DocumentModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    # pylint: disable=not-callable
    original_created_time = Column(DateTime(timezone=True)) # see sqlalchemy datetime info here: https://stackoverflow.com/questions/13370317/sqlalchemy-default-datetime
    text = Column(String)
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
    embedding_computations = relationship("DocumentEmbeddingComputationModel", back_populates="document")
    used_in_trained_models = relationship("BertopicTrainedModel", secondary="documentbertopictrainedmodel", back_populates="trained_on_documents")
    mattermost_document = relationship("MattermostDocumentModel", primaryjoin="DocumentModel.id==MattermostDocumentModel.document")
