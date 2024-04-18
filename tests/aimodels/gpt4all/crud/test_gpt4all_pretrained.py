from sqlalchemy.orm import Session
from ppg.schemas.gpt4all.llm_pretrained import LlmPretrainedCreate
from app.aimodels.gpt4all.crud.crud_llm_pretrained import llm_pretrained
from app.aimodels.gpt4all.models.llm_pretrained import LlmPretrainedModel

# assert llm_pretrained was built with correct model
def test_llm_pretrained():
    assert llm_pretrained.model == LlmPretrainedModel

# get_by_sha256 with existing sha256
def test_get_by_sha256_existing_sha256(db: Session, valid_sha256: str):
    # create a llm_pretrained
    embedding_pretrained_create = LlmPretrainedCreate(sha256=valid_sha256, model_name='test')
    llm_pretrained.create(db, obj_in=embedding_pretrained_create)

    # get object from db
    llm_pretrained_object = llm_pretrained.get_by_sha256(db, sha256=valid_sha256)

    assert llm_pretrained_object.sha256 == valid_sha256

# get_by_sha256 with non-existing sha256
def test_get_by_sha256_non_existing_sha256(db: Session, valid_sha256: str):
    # get object from db
    llm_pretrained_object = llm_pretrained.get_by_sha256(db, sha256=valid_sha256)

    assert llm_pretrained_object is None

# get_by_sha256 with empty sha256
def test_get_by_sha256_empty_or_none_sha256(db: Session):
    # get object from db
    llm_pretrained_object = llm_pretrained.get_by_sha256(db, sha256='')
    assert llm_pretrained_object is None

    llm_pretrained_object = llm_pretrained.get_by_sha256(db, sha256=None)
    assert llm_pretrained_object is None
