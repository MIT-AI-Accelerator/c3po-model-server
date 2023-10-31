import uuid
import enum
from sqlalchemy import Column, UUID, Enum, String
from app.db.base_class import Base
from app.core.config import OriginationEnum, get_originated_from

class BertopicVisualizationTypeEnum(str, enum.Enum):
    MODEL_CLUSTERS = "model_topic_clusters"
    MODEL_WORDS = "model_topic_words"
    MODEL_TIMELINE = "model_topic_timeline"
    TOPIC_TIMELINE = "topic_timeline"

class BertopicVisualizationModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    model_or_topic_id = Column(UUID)
    visualization_type = Column(Enum(BertopicVisualizationTypeEnum), nullable=False)
    html_string = Column(String)
    json_string = Column(String)
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
