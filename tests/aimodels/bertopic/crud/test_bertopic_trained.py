from app.aimodels.bertopic.crud.crud_bertopic_trained import bertopic_trained
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel
from sqlalchemy.orm import Session
from app.aimodels.bertopic.crud.crud_bertopic_embedding_pretrained import bertopic_embedding_pretrained
from app.aimodels.bertopic.schemas.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate
from app.aimodels.bertopic.schemas.bertopic_trained import BertopicTrainedCreate
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import EmbeddingModelTypeEnum

# assert bertopic_trained was built with correct model
def test_bertopic_trained():
    assert bertopic_trained.model == BertopicTrainedModel

def test_create_with_bad_embedding_pretrained_id(db: Session):
    obj_in = BertopicTrainedCreate()
    obj = bertopic_trained.create_with_embedding_pretrained_id(db, obj_in=obj_in, embedding_pretrained_id="00000000-0000-0000-0000-000000000000")
    assert obj is None

def test_create_with_good_embedding_pretrained_id(db: Session, valid_sha256: str):
    obj_in = BertopicTrainedCreate()

    # create a bertopic_embedding_pretrained sentence transformer
    embedding_pretrained_create = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256, model_name='test')
    embedding_pretrained_obj = bertopic_embedding_pretrained.create(db, obj_in=embedding_pretrained_create)

    # create a bertopic_trained with the bertopic_embedding_pretrained id
    obj = bertopic_trained.create_with_embedding_pretrained_id(db, obj_in=obj_in, embedding_pretrained_id=embedding_pretrained_obj.id)

    assert obj is not None
    assert obj.embedding_pretrained_id == embedding_pretrained_obj.id
    assert obj.embedding_pretrained.sha256 == embedding_pretrained_obj.sha256
    assert obj.uploaded == False

    # create a bertopic_embedding_pretrained weak learner
    embedding_pretrained_create = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256, model_name='test',
                                                                    model_type=EmbeddingModelTypeEnum.WEAK_LEARNERS)
    embedding_pretrained_obj = bertopic_embedding_pretrained.create(db, obj_in=embedding_pretrained_create)

    # create a bertopic_trained with the bertopic_embedding_pretrained id
    obj = bertopic_trained.create_with_embedding_pretrained_id(db, obj_in=obj_in, embedding_pretrained_id=embedding_pretrained_obj.id)

    assert obj is not None
    assert obj.embedding_pretrained_id == embedding_pretrained_obj.id
    assert obj.embedding_pretrained.sha256 == embedding_pretrained_obj.sha256
    assert obj.embedding_pretrained.model_type == embedding_pretrained_obj.model_type
    assert obj.uploaded == False

# test that given two bertopic_trained objects with the same embedding_pretrained_id, when we get_multi_by_embedding_pretrained_id, we get both
def test_get_multi_by_embedding_pretrained_id(db: Session, valid_sha256: str):
    obj = BertopicTrainedCreate()
    obj2 = BertopicTrainedCreate()

    # create a bertopic_embedding_pretrained
    embedding_pretrained_create = BertopicEmbeddingPretrainedCreate(sha256=valid_sha256, model_name='test')
    embedding_pretrained_obj = bertopic_embedding_pretrained.create(db, obj_in=embedding_pretrained_create)

    # create a bertopic_trained with the bertopic_embedding_pretrained id for each bertopic_trained
    obj = bertopic_trained.create_with_embedding_pretrained_id(db, obj_in=obj, embedding_pretrained_id=embedding_pretrained_obj.id)
    obj2 = bertopic_trained.create_with_embedding_pretrained_id(db, obj_in=obj2, embedding_pretrained_id=embedding_pretrained_obj.id)

    # get all bertopic_trained with the same embedding_pretrained_id
    objs = bertopic_trained.get_multi_by_embedding_pretrained_id(db, embedding_pretrained_id=embedding_pretrained_obj.id)

    assert objs is not None
    assert len(objs) == 2
    assert objs[0].embedding_pretrained_id == embedding_pretrained_obj.id
    assert objs[1].embedding_pretrained_id == embedding_pretrained_obj.id

# test that nothing is returned when we get_multi_by_embedding_pretrained_id with a bad id
def test_get_multi_by_embedding_pretrained_id_with_bad_id(db: Session):
    objs = bertopic_trained.get_multi_by_embedding_pretrained_id(db, embedding_pretrained_id="00000000-0000-0000-0000-000000000000")
    assert objs is not None
    assert len(objs) == 0
