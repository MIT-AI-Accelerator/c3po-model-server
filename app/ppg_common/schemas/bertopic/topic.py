from pydantic import BaseModel, UUID4, ConfigDict
from typing import Optional
from app.core.config import OriginationEnum

# Shared properties
class TopicSummaryBase(BaseModel):
    topic_id: Optional[int] = None
    name: Optional[str] = None
    top_n_words: Optional[str] = None
    top_n_documents: Optional[dict] = None
    summary: Optional[str] = None
    is_trending: bool = False

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Properties to receive on TopicSummary creation
class TopicSummaryCreate(TopicSummaryBase):
    pass

# Properties to receive on TopicSummary update
class TopicSummaryUpdate(TopicSummaryBase):
    model_id: UUID4

# Properties shared by models stored in DB
class TopicSummaryInDBBase(TopicSummaryBase):
    id: UUID4
    originated_from: OriginationEnum

# Properties to return to client
class TopicSummary(TopicSummaryInDBBase):
    pass

# Properties properties stored in DB
class TopicSummaryInDB(TopicSummaryInDBBase):
    pass
