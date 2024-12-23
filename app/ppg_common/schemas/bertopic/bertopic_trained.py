from typing import Optional
from datetime import datetime
from pydantic import BaseModel, UUID4, ConfigDict
from app.core.config import OriginationEnum
from .bertopic_embedding_pretrained import BertopicEmbeddingPretrained
from .document import Document

# Shared properties
class BertopicTrainedBase(BaseModel):
    uploaded: Optional[bool] = None
    sentence_transformer_id: Optional[UUID4] = None
    weak_learner_id: Optional[UUID4] = None
    summarization_model_id: Optional[UUID4] = None
    seed_topics: Optional[dict] = None
    stop_words: Optional[dict] = None
    prompt_template: Optional[str] = None
    refine_template: Optional[str] = None

# Properties to receive on BertopicTrained creation
class BertopicTrainedCreate(BertopicTrainedBase):
    uploaded: bool = False

# Properties to receive on BertopicTrained update
class BertopicTrainedUpdate(BertopicTrainedBase):
    pass

# Properties shared by models stored in DB
class BertopicTrainedInDBBase(BertopicTrainedBase):
    id: UUID4
    time: datetime
    embedding_pretrained_id: UUID4
    originated_from: OriginationEnum

    model_config = ConfigDict(from_attributes=True)

# Properties to return to client
class BertopicTrained(BertopicTrainedInDBBase):
    embedding_pretrained: BertopicEmbeddingPretrained
    trained_on_documents: list[Document]

# Properties properties stored in DB
class BertopicTrainedInDB(BertopicTrainedInDBBase):
    pass
