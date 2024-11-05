import uuid
from typing import TYPE_CHECKING
from sqlalchemy import Column, Enum, Integer, UUID, String, Boolean, Sequence
from sqlalchemy.orm import relationship
from app.db.base_class import Base
from app.core.config import get_originated_from, OriginationEnum
from app.ppg_common.schemas.gpt4all.llm_pretrained import LlmFilenameEnum

# add in the TYPE_CHECKING check here if relationships are created

class LlmPretrainedModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    model_type = Column(Enum(LlmFilenameEnum), default=LlmFilenameEnum.L13B_SNOOZY)
    version_sequence = Sequence(__qualname__.lower() + "_version_sequence") # see here for autoincrementing versioning: https://copyprogramming.com/howto/using-sqlalchemy-orm-for-a-non-primary-key-unique-auto-incrementing-id
    version = Column(Integer, version_sequence, server_default=version_sequence.next_value(), index=True, unique=True, nullable=False)
    sha256 = Column(String(64))
    uploaded = Column(Boolean(), default=False)
    use_base_model = Column(Boolean(), default=False)
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
