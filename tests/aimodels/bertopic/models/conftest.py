import uuid
from sqlalchemy.orm import Session
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.models.document_embedding_computation import DocumentEmbeddingComputationModel
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel
from collections.abc import Generator
import pytest

@pytest.fixture(scope="module")
def obj_ids(db: Session) -> Generator:
    embedding_pretrained_id = uuid.uuid4()
    document_id = uuid.uuid4()
    embedding_computation_id = uuid.uuid4()
    bertopic_trained_id = uuid.uuid4()

    obj_id_dict = {
        "embedding_pretrained_id": embedding_pretrained_id,
        "document_id": document_id,
        "embedding_computation_id": embedding_computation_id,
        "trained_id": bertopic_trained_id,
    }

    # build base models
    embedding_pretrained_db = BertopicEmbeddingPretrainedModel(
        id=embedding_pretrained_id,
    )

    document_db = DocumentModel(
        id=document_id,
    )

    # build the relationships
    embedding_computation_db = DocumentEmbeddingComputationModel(
        id=embedding_computation_id,
        document_id=document_id,
        bertopic_embedding_pretrained_id=embedding_pretrained_id,
    )

    trained_db = BertopicTrainedModel(
        id=bertopic_trained_id,
        embedding_pretrained_id=embedding_pretrained_id,
        trained_on_documents=[document_db],
    )

    # add the relationships
    db.add_all([embedding_pretrained_db, document_db, embedding_computation_db, trained_db])
    db.commit()

    return obj_id_dict
