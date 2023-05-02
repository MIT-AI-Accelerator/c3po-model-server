

from sqlalchemy.orm import Session
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.models.document_embedding_computation import DocumentEmbeddingComputationModel
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel


def test_instantiation(db: Session, obj_ids: dict):
    document_db = db.query(DocumentModel).filter(
        DocumentModel.id == obj_ids["document_id"]).first()

    assert document_db is not None

def test_relationships(db: Session, obj_ids: dict):
    document_id = obj_ids["document_id"]
    embedding_computation_id = obj_ids["embedding_computation_id"]
    trained_id = obj_ids["trained_id"]

    # build base models
    document_db = db.query(DocumentModel).filter(
        DocumentModel.id == document_id).first()

    embedding_computation_db = db.query(DocumentEmbeddingComputationModel).filter(
        DocumentEmbeddingComputationModel.id == embedding_computation_id).first()

    trained_db = db.query(BertopicTrainedModel).filter(
        BertopicTrainedModel.id == trained_id).first()

    # test that the relationships are correctly loaded
    assert document_db.embedding_computations == [embedding_computation_db]
    assert document_db.used_in_trained_models == [trained_db]
