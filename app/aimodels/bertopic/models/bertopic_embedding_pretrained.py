import enum
from typing import TYPE_CHECKING
from sqlalchemy import Column, Enum, Integer, UUID, String, Boolean, Sequence
from sqlalchemy.orm import relationship
from app.db.base_class import Base
import uuid

if TYPE_CHECKING:
    from .document_embedding_computation import DocumentEmbeddingComputationModel  # noqa: F401
    from .bertopic_trained import BertopicTrainedModel  # noqa: F401

class EmbeddingModelTypeEnum(str, enum.Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    DIFF_CSE = "diff_cse"

class BertopicEmbeddingPretrainedModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    model_type = Column(Enum(EmbeddingModelTypeEnum), default=EmbeddingModelTypeEnum.SENTENCE_TRANSFORMERS)
    version_sequence = Sequence(__qualname__.lower() + "_version_sequence") # see here for autoincrementing versioning: https://copyprogramming.com/howto/using-sqlalchemy-orm-for-a-non-primary-key-unique-auto-incrementing-id
    version = Column(Integer, version_sequence, server_default=version_sequence.next_value(), index=True, unique=True, nullable=False)
    sha256 = Column(String(64))
    uploaded = Column(Boolean(), default=False)
    document_embedding_computations = relationship("DocumentEmbeddingComputationModel", back_populates="bertopic_embedding_pretrained")
    trained_models = relationship("BertopicTrainedModel", back_populates="embedding_pretrained")
