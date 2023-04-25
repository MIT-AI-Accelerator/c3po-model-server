from typing import Optional

from pydantic import BaseModel, UUID4

from .bertopic_embedding_pretrained import BertopicEmbeddingPretrained
from .document import Document

# Shared properties
class BertopicTrainedBase(BaseModel):
    uploaded: Optional[bool] = None
    plotly_bubble_config: Optional[dict] = None

# Properties to receive on BertopicTrained creation
class BertopicTrainedCreate(BertopicTrainedBase):
    uploaded: bool = False

# Properties to receive on BertopicTrained update
class BertopicTrainedUpdate(BertopicTrainedBase):
    pass

# Properties shared by models stored in DB
class BertopicTrainedInDBBase(BertopicTrainedBase):
    id: UUID4
    embedding_pretrained_id: UUID4

    class Config:
        orm_mode = True

# Properties to return to client
class BertopicTrained(BertopicTrainedInDBBase):
    embedding_pretrained: BertopicEmbeddingPretrained
    trained_on_documents: list[Document]

# Properties properties stored in DB
class BertopicTrainedInDB(BertopicTrainedInDBBase):
    pass
