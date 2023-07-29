from app.crud.base import CRUDBase
from ..models.topic import TopicSummaryModel
from ..schemas.topic import TopicSummaryCreate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDTopicSummary(CRUDBase[TopicSummaryModel, TopicSummaryCreate, TopicSummaryCreate]):
    pass

topic_summary = CRUDTopicSummary(TopicSummaryModel)
