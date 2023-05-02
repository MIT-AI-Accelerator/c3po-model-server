

from sqlalchemy.orm import Session
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.models.document_embedding_computation import DocumentEmbeddingComputationModel
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel


def test_instantiation(db: Session, obj_ids: dict):
    document_embedding_computation_id = obj_ids["embedding_computation_id"]
    document_embedding_computation_db = db.query(DocumentEmbeddingComputationModel).filter(
        DocumentEmbeddingComputationModel.id == document_embedding_computation_id).first()

    assert document_embedding_computation_db is not None

def test_relationships(db: Session, obj_ids: dict):
    document_embedding_computation_id = obj_ids["embedding_computation_id"]
    document_id = obj_ids["document_id"]
    bertopic_embedding_pretrained_id = obj_ids["embedding_pretrained_id"]

    # build base models
    document_embedding_computation_db = db.query(DocumentEmbeddingComputationModel).filter(
        DocumentEmbeddingComputationModel.id == document_embedding_computation_id).first()

    document_db = db.query(DocumentModel).filter(
        DocumentModel.id == document_id).first()

    bertopic_embedding_pretrained_db = db.query(BertopicEmbeddingPretrainedModel).filter(
        BertopicEmbeddingPretrainedModel.id == bertopic_embedding_pretrained_id).first()

    # test that the relationships are correctly loaded
    assert document_embedding_computation_db.document == document_db
    assert document_embedding_computation_db.bertopic_embedding_pretrained == bertopic_embedding_pretrained_db
