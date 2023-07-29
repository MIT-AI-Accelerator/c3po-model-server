from pydantic import BaseModel, UUID4
from typing import Optional
from app.core.config import OriginationEnum
# from ..models.topic import TopicDocument

# Shared properties
class TopicSummaryBase(BaseModel):
    topic_id: int
    name: str
    top_n_words: str = ""
    # documents: list[TopicSummaryDocument]
    summary: str

# Properties to receive on TopicSummary creation
class TopicSummaryCreate(TopicSummaryBase):
    pass

# Properties to receive on TopicSummary update
class TopicSummaryUpdate(TopicSummaryBase):
    model_id: str

# Properties shared by models stored in DB
class TopicSummaryInDBBase(TopicSummaryBase):
    id: UUID4
    originated_from: OriginationEnum

    class Config:
        orm_mode = True

# Properties to return to client
class TopicSummary(TopicSummaryInDBBase):
    pass

# Properties properties stored in DB
class TopicSummaryInDB(TopicSummaryInDBBase):
    pass
