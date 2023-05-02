from sqlalchemy.orm import Session
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel


def test_instantiation(db: Session, obj_ids: dict):
    # test that the model can be instantiated and saved to the database
    trained_model = db.query(BertopicTrainedModel).filter(
        BertopicTrainedModel.id == obj_ids["trained_id"]).first()

    assert trained_model is not None
    assert trained_model.uploaded is False


def test_relationships(db: Session, obj_ids: dict):

        embedding_pretrained_id = obj_ids["embedding_pretrained_id"]
        document_id = obj_ids["document_id"]
        trained_id = obj_ids["trained_id"]

        # build base models
        embedding_pretrained_db = db.query(BertopicEmbeddingPretrainedModel).filter(
            BertopicEmbeddingPretrainedModel.id == embedding_pretrained_id).first()
        document_db = db.query(DocumentModel).filter(
            DocumentModel.id == document_id).first()
        trained_db = db.query(BertopicTrainedModel).filter(
            BertopicTrainedModel.id == trained_id).first()

        # test that the relationships are correctly loaded
        assert trained_db.embedding_pretrained == embedding_pretrained_db
        assert trained_db.embedding_pretrained_id == embedding_pretrained_id
        assert trained_db.trained_on_documents == [document_db]
