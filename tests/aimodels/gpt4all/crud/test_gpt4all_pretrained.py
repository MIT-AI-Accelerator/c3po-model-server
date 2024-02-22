from sqlalchemy.orm import Session
from ppg.schemas.gpt4all.gpt4all_pretrained import Gpt4AllPretrainedCreate
from app.aimodels.gpt4all.crud.crud_gpt4all_pretrained import gpt4all_pretrained
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllPretrainedModel

# assert gpt4all_pretrained was built with correct model
def test_gpt4all_pretrained():
    assert gpt4all_pretrained.model == Gpt4AllPretrainedModel

# get_by_sha256 with existing sha256
def test_get_by_sha256_existing_sha256(db: Session, valid_sha256: str):
    # create a gpt4all_pretrained
    embedding_pretrained_create = Gpt4AllPretrainedCreate(sha256=valid_sha256, model_name='test')
    gpt4all_pretrained.create(db, obj_in=embedding_pretrained_create)

    # get object from db
    gpt4all_pretrained_object = gpt4all_pretrained.get_by_sha256(db, sha256=valid_sha256)

    assert gpt4all_pretrained_object.sha256 == valid_sha256

# get_by_sha256 with non-existing sha256
def test_get_by_sha256_non_existing_sha256(db: Session, valid_sha256: str):
    # get object from db
    gpt4all_pretrained_object = gpt4all_pretrained.get_by_sha256(db, sha256=valid_sha256)

    assert gpt4all_pretrained_object is None

# get_by_sha256 with empty sha256
def test_get_by_sha256_empty_or_none_sha256(db: Session):
    # get object from db
    gpt4all_pretrained_object = gpt4all_pretrained.get_by_sha256(db, sha256='')
    assert gpt4all_pretrained_object is None

    gpt4all_pretrained_object = gpt4all_pretrained.get_by_sha256(db, sha256=None)
    assert gpt4all_pretrained_object is None
