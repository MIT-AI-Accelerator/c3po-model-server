import enum
from pydantic import BaseModel, UUID4, ConfigDict
from app.core.config import OriginationEnum


class BertopicVisualizationTypeEnum(str, enum.Enum):
    MODEL_CLUSTERS = "model_topic_clusters"
    MODEL_WORDS = "model_topic_words"
    MODEL_TIMELINE = "model_topic_timeline"
    TOPIC_TIMELINE = "topic_timeline"

# Shared properties
class BertopicVisualizationBase(BaseModel):
    model_or_topic_id: UUID4
    visualization_type: BertopicVisualizationTypeEnum
    html_string: str
    json_string: str

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Properties to receive on BertopicVisualization creation
class BertopicVisualizationCreate(BertopicVisualizationBase):
    pass

# Properties shared by models stored in DB
class BertopicVisualizationInDBBase(BertopicVisualizationBase):
    id: UUID4
    originated_from: OriginationEnum

# Properties to return to client
class BertopicVisualization(BertopicVisualizationInDBBase):
    pass

# Properties properties stored in DB
class BertopicVisualizationInDB(BertopicVisualizationInDBBase):
    pass
