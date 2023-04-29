from app.aimodels.bertopic.crud.crud_document_embedding_computation import document_embedding_computation
from app.aimodels.bertopic.models.document_embedding_computation import DocumentEmbeddingComputationModel

# assert document_embedding_computation was built with correct model
def test_document_embedding_computation():
    assert document_embedding_computation.model == DocumentEmbeddingComputationModel
