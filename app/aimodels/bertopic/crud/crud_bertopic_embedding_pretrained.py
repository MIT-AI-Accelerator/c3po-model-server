
from app.crud.base import CRUDBase
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from ..schemas.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDBertopicEmbeddingPretrained(CRUDBase[BertopicEmbeddingPretrainedModel, BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate]  ):
    pass

bertopic_embedding_pretrained = CRUDBertopicEmbeddingPretrained(BertopicEmbeddingPretrainedModel)
