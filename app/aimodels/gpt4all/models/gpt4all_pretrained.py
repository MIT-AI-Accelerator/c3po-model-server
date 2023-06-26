import enum
from typing import TYPE_CHECKING
from sqlalchemy import Column, Enum, Integer, UUID, String, Boolean, Sequence
from sqlalchemy.orm import relationship
from app.db.base_class import Base
import uuid

# add in the TYPE_CHECKING check here if relationships are created

class Gpt4AllModelFilenameEnum(str, enum.Enum):
    L13B_SNOOZY = "ggml-gpt4all-l13b-snoozy.bin"

class Gpt4AllPretrainedModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    model_type = Column(Enum(Gpt4AllModelFilenameEnum), default=Gpt4AllModelFilenameEnum.L13B_SNOOZY)
    version_sequence = Sequence(__qualname__.lower() + "_version_sequence") # see here for autoincrementing versioning: https://copyprogramming.com/howto/using-sqlalchemy-orm-for-a-non-primary-key-unique-auto-incrementing-id
    version = Column(Integer, version_sequence, server_default=version_sequence.next_value(), index=True, unique=True, nullable=False)
    sha256 = Column(String(64))
    uploaded = Column(Boolean(), default=False)
    use_base_model = Column(Boolean(), default=False)
