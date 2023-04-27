# this is a basic model file for "Test" entity
from sqlalchemy import Column, UUID, Boolean, String
from app.db.base_class import Base
import uuid

class TestModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    bool_field = Column(Boolean(), default=False)
    title = Column(String(), default="title")
