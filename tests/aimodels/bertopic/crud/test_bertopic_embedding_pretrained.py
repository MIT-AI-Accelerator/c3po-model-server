from sqlalchemy.orm import Session
from ppg.schemas.bertopic.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate
from app.aimodels.bertopic.crud.crud_bertopic_embedding_pretrained import bertopic_embedding_pretrained
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel


# assert bertopic_embedding_pretrained was built with correct model
def test_bertopic_embedding_pretrained():
    assert bertopic_embedding_pretrained.model == BertopicEmbeddingPretrainedModel

# get_by_sha256 with existing sha256
def test_get_by_sha256_existing_sha256(db: Session, valid_sha256: str):
    # create a bertopic_embedding_pretrained
    embedding_pretrained_create = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256, model_name='test')
    bertopic_embedding_pretrained.create(db, obj_in=embedding_pretrained_create)

    # get object from db
    bertopic_embedding_pretrained_object = bertopic_embedding_pretrained.get_by_sha256(db, sha256=valid_sha256)

    assert bertopic_embedding_pretrained_object.sha256 == valid_sha256

# get_by_sha256 with non-existing sha256
def test_get_by_sha256_non_existing_sha256(db: Session, valid_sha256: str):
    # get object from db
    bertopic_embedding_pretrained_object = bertopic_embedding_pretrained.get_by_sha256(db, sha256=valid_sha256)

    assert bertopic_embedding_pretrained_object is None

# get_by_sha256 with empty sha256
def test_get_by_sha256_empty_or_none_sha256(db: Session):
    # get object from db
    bertopic_embedding_pretrained_object = bertopic_embedding_pretrained.get_by_sha256(db, sha256='')
    assert bertopic_embedding_pretrained_object is None

    bertopic_embedding_pretrained_object = bertopic_embedding_pretrained.get_by_sha256(db, sha256=None)
    assert bertopic_embedding_pretrained_object is None
