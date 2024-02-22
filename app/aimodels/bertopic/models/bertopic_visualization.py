import uuid
from sqlalchemy import Column, UUID, Enum, String
from ppg.core.config import OriginationEnum
from ppg.schemas.bertopic.bertopic_visualization import BertopicVisualizationTypeEnum
from app.db.base_class import Base
from app.core.config import get_originated_from


class BertopicVisualizationModel(Base):
    id = Column(UUID, primary_key=True, unique=True, default=uuid.uuid4)
    model_or_topic_id = Column(UUID)
    visualization_type = Column(Enum(BertopicVisualizationTypeEnum), nullable=False)
    html_string = Column(String)
    json_string = Column(String)
    originated_from = Column(Enum(OriginationEnum), default=get_originated_from)
