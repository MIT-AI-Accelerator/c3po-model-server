from app.aimodels.bertopic.crud.crud_bertopic_embedding_pretrained import bertopic_embedding_pretrained
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel

# assert bertopic_embedding_pretrained was built with correct model
def test_bertopic_embedding_pretrained():
    assert bertopic_embedding_pretrained.model == BertopicEmbeddingPretrainedModel
