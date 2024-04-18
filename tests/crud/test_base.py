from sqlalchemy.orm import Session
import uuid
from tests.test_files import crud
from tests.test_files.crud.crud_test_model import EmptyTestModel
from tests.test_files.crud.crud_test_schema import EmptyTestCreate, EmptyTestUpdate
from fastapi.encoders import jsonable_encoder

def test_create_obj(db: Session) -> None:
    obj_in = EmptyTestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)
    assert obj.bool_field is False
    assert obj.title == "title"
    assert isinstance(obj.id, uuid.UUID)

def test_get_obj(db: Session) -> None:
    obj_in = EmptyTestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)
    stored_obj = crud.crud_test.get(db=db, id=obj.id)
    assert stored_obj
    assert obj.id == stored_obj.id
    assert obj.bool_field == stored_obj.bool_field
    assert obj.title == stored_obj.title

def test_get_obj_bad_id(db: Session) -> None:
    stored_obj = crud.crud_test.get(db=db, id=uuid.uuid4())
    assert stored_obj is None

def test_get_multi(db: Session) -> None:
    obj_in = EmptyTestCreate()
    obj_in_2 = EmptyTestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)
    obj_2 = crud.crud_test.create(db=db, obj_in=obj_in_2)
    all_objs = crud.crud_test.get_multi(db=db, skip=0, limit=10000)

    assert obj in all_objs
    assert obj_2 in all_objs

def test_update_obj(db: Session) -> None:
    obj_in = EmptyTestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)

    # update the obj
    new_bool_field = True
    obj_update = EmptyTestUpdate(bool_field=new_bool_field)
    crud.crud_test.update(db=db, db_obj=obj, obj_in=obj_update)

    # get the obj again
    obj2 = crud.crud_test.get(db=db, id=obj.id)

    # assert objs are same id but bool_field updated, and other fields unchanged
    assert obj.id == obj2.id
    assert obj2.bool_field == new_bool_field
    assert obj.title == obj2.title

def test_update_obj_bad_id(db: Session) -> None:
    obj = EmptyTestModel(id=uuid.uuid4(), bool_field=False, title="title")

    # update the obj that doesn't exist
    obj_update = EmptyTestUpdate(bool_field=True)
    crud.crud_test.update(db=db, db_obj=obj, obj_in=obj_update)

    # get the obj again
    obj2 = crud.crud_test.get(db=db, id=obj.id)

    # assert objs are same id but bool_field updated, and other fields unchanged
    assert obj2 is None

# given two objects in the same table, when the second is created, assert json encoding is empty
# then assert that updating refreshes the first object by changing a field
def test_update_obj_refreshes_if_expired(db: Session) -> None:
    obj_in = EmptyTestCreate()
    obj_in_2 = EmptyTestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)

    # object created just fine
    obj_data = jsonable_encoder(obj)
    assert 'bool_field' in obj_data

    # create another object
    crud.crud_test.create(db=db, obj_in=obj_in_2)

    # original object has been expired by sqlalchemy, so json encoding is empty
    obj_data = jsonable_encoder(obj)
    assert 'bool_field' not in obj_data

    # update the obj
    new_bool_field = True
    new_title = "t2"
    obj_update = EmptyTestUpdate(bool_field=new_bool_field, title=new_title)
    crud.crud_test.update(db=db, db_obj=obj, obj_in=obj_update)

    # get the obj again
    new_obj = crud.crud_test.get(db=db, id=obj.id)

    # assert objs are same id and both original db object and new one have been refreshed
    assert obj.id == new_obj.id
    assert new_obj.bool_field == new_bool_field
    assert new_obj.title == new_title
    assert obj.title == "t2"

def test_update_obj_dict(db: Session) -> None:
    obj_in = EmptyTestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)

    # update the obj
    obj_dict = {'bool_field': True, 'title': 'test'}
    crud.crud_test.update(db=db, db_obj=obj, obj_in=obj_dict)

    obj_updated = crud.crud_test.get(db=db, id=obj.id)
    assert obj_updated.id == obj.id
    assert obj_updated.bool_field == obj_dict['bool_field']
    assert obj_updated.title == obj_dict['title']

def test_delete_obj(db: Session) -> None:
    obj_in = EmptyTestCreate()

    # create, delete, and get obj
    obj = crud.crud_test.create(db=db, obj_in=obj_in)
    obj2 = crud.crud_test.remove(db=db, id=obj.id)
    obj3 = crud.crud_test.get(db=db, id=obj.id)

    assert obj3 is None
    assert obj2.id == obj.id
    assert obj2.title == "title"
    assert obj2.bool_field is False

def test_delete_obj_bad_id(db: Session) -> None:
    obj = crud.crud_test.remove(db=db, id=uuid.uuid4())
    assert obj is None

# given three objects, update two of them, then refresh all three, assert that the two updated objects are updated and the third is unchanged
def test_refresh_all_by_id(db: Session) -> None:
    obj_in_1 = EmptyTestCreate()
    obj_in_2 = EmptyTestCreate()
    obj_in_3 = EmptyTestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in_1)
    obj2 = crud.crud_test.create(db=db, obj_in=obj_in_2)
    obj3 = crud.crud_test.create(db=db, obj_in=obj_in_3)

    # assert obj and obj2 are empty now that obj3 is created, happens D2 sqlalchemy expiration
    assert not jsonable_encoder(obj)
    assert not jsonable_encoder(obj2)

    # update obj2 and 3
    new_bool_field = True
    obj2_update = EmptyTestUpdate(bool_field=new_bool_field)
    obj3_update = EmptyTestUpdate(bool_field=new_bool_field)

    # update and assert that fields were refreshed appropriately
    crud.crud_test.update(db=db, db_obj=obj2, obj_in=obj2_update)
    assert 'bool_field' in jsonable_encoder(obj2)

    # update and assert that fields were refreshed appropriately but obj2 expired
    crud.crud_test.update(db=db, db_obj=obj3, obj_in=obj3_update)
    assert not jsonable_encoder(obj2)
    assert 'bool_field' in jsonable_encoder(obj3)

    # refresh all objects
    objs = crud.crud_test.refresh_all_by_id(db=db, db_obj_ids=[obj.id, obj2.id, obj3.id])

    # assert objs contains original objects and that the fields are appropriately updated
    indexed_obj_0 = [item for item in objs if item.id == obj.id][0]
    indexed_obj_2 = [item for item in objs if item.id == obj2.id][0]
    indexed_obj_3 = [item for item in objs if item.id == obj3.id][0]

    assert indexed_obj_0.bool_field is False
    assert indexed_obj_2.bool_field == new_bool_field
    assert indexed_obj_3.bool_field == new_bool_field
    assert indexed_obj_0.title == 'title'
    assert indexed_obj_2.title == 'title'
    assert indexed_obj_3.title == 'title'

def test_refresh_all_by_id_empty_or_none_id_list(db: Session) -> None:
    # refresh all objects with empty list
    objs = crud.crud_test.refresh_all_by_id(db=db, db_obj_ids=[])
    assert objs == []

    # refresh all objects with None
    objs = crud.crud_test.refresh_all_by_id(db=db, db_obj_ids=None)
    assert objs == []

# given three objects to create, create them and assert they are not None and have the correct custom fields
def test_create_all_using_id(db: Session) -> None:
    obj_in_list = [EmptyTestCreate(title="1st"), EmptyTestCreate(title="2nd"), EmptyTestCreate(title="3rd")]
    objs = crud.crud_test.create_all_using_id(db=db, obj_in_list=obj_in_list)

    # assert objs are not None
    assert objs[0] is not None and objs[0].title == "1st"
    assert objs[1] is not None and objs[1].title == "2nd"
    assert objs[2] is not None and objs[2].title == "3rd"

def test_create_all_using_id_empty_or_none_list(db: Session) -> None:
    # create all objects with empty list
    objs = crud.crud_test.create_all_using_id(db=db, obj_in_list=[])
    assert objs == []

    # create all objects with None
    objs = crud.crud_test.create_all_using_id(db=db, obj_in_list=None)
    assert objs == []
