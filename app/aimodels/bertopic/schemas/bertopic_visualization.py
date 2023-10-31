from pydantic import BaseModel, UUID4
from app.core.config import OriginationEnum
from ..models.bertopic_visualization import BertopicVisualizationTypeEnum

# Shared properties
class BertopicVisualizationBase(BaseModel):
    model_or_topic_id: UUID4
    visualization_type: BertopicVisualizationTypeEnum
    html_string: str
    json_string: str


# Properties to receive on BertopicVisualization creation
class BertopicVisualizationCreate(BertopicVisualizationBase):
    pass

# Properties shared by models stored in DB
class BertopicVisualizationInDBBase(BertopicVisualizationBase):
    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class BertopicVisualization(BertopicVisualizationInDBBase):
    pass

# Properties properties stored in DB
class BertopicVisualizationInDB(BertopicVisualizationInDBBase):
    pass
