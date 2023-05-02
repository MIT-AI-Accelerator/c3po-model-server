from sqlalchemy.orm import Session
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.aimodels.bertopic.models.document_embedding_computation import DocumentEmbeddingComputationModel
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel


def test_instantiation(db: Session, obj_ids: dict):
    # test that the model can be instantiated and saved to the database
    embedding_model = db.query(BertopicEmbeddingPretrainedModel).filter(
        BertopicEmbeddingPretrainedModel.id == obj_ids["embedding_pretrained_id"]).first()

    assert embedding_model is not None
    assert embedding_model.version is not None


def test_relationships(db: Session, obj_ids: dict):

    embedding_pretrained_id = obj_ids["embedding_pretrained_id"]
    embedding_computation_id = obj_ids["embedding_computation_id"]

    # build base models
    embedding_pretrained_db = db.query(BertopicEmbeddingPretrainedModel).filter(
        BertopicEmbeddingPretrainedModel.id == embedding_pretrained_id).first()
    embedding_computation_db = db.query(DocumentEmbeddingComputationModel).filter(
        DocumentEmbeddingComputationModel.id == embedding_computation_id).first()
    trained_db = db.query(BertopicTrainedModel).filter(
        BertopicTrainedModel.embedding_pretrained_id == embedding_pretrained_id).first()

    # test that the relationships are correctly loaded
    assert embedding_pretrained_db.document_embedding_computations == [
        embedding_computation_db]
    assert embedding_pretrained_db.trained_models == [trained_db]
