from sqlalchemy.orm import Session
import uuid
from tests.test_files import crud
from tests.test_files.crud.crud_test_schema import TestCreate, TestUpdate
from fastapi.encoders import jsonable_encoder

def test_create_obj(db: Session) -> None:
    obj_in = TestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)
    assert obj.bool_field is False
    assert obj.title == "title"
    assert isinstance(obj.id, uuid.UUID)

def test_get_obj(db: Session) -> None:
    obj_in = TestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)
    stored_obj = crud.crud_test.get(db=db, id=obj.id)
    assert stored_obj
    assert obj.id == stored_obj.id
    assert obj.bool_field == stored_obj.bool_field
    assert obj.title == stored_obj.title


def test_update_obj(db: Session) -> None:
    obj_in = TestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in)

    # update the obj
    new_bool_field = True
    obj_update = TestUpdate(bool_field=new_bool_field)
    obj2 = crud.crud_test.update(db=db, db_obj=obj, obj_in=obj_update)

    # assert objs are same id but bool_field updated, and other fields unchanged
    assert obj.id == obj2.id
    assert obj2.bool_field == new_bool_field
    assert obj.title == obj2.title

# given two objects in the same table, when the second is created, assert json encoding is empty
# then assert that updating refreshes the first object by changing a field
def test_update_obj_refreshes_if_expired(db: Session) -> None:
    obj_in = TestCreate()
    obj_in_2 = TestCreate()
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
    obj_update = TestUpdate(bool_field=new_bool_field, title=new_title)
    new_obj = crud.crud_test.update(db=db, db_obj=obj, obj_in=obj_update)

    # assert objs are same id and both original db object and new one have been refreshed
    assert obj.id == new_obj.id
    assert new_obj.bool_field == new_bool_field
    assert new_obj.title == new_title
    assert obj.title == "t2"


def test_delete_obj(db: Session) -> None:
    obj_in = TestCreate()

    # create, delete, and get obj
    obj = crud.crud_test.create(db=db, obj_in=obj_in)
    obj2 = crud.crud_test.remove(db=db, id=obj.id)
    obj3 = crud.crud_test.get(db=db, id=obj.id)

    assert obj3 is None
    assert obj2.id == obj.id
    assert obj2.title == "title"
    assert obj2.bool_field is False

# given three objects, update two of them, then refresh all three, assert that the two updated objects are updated and the third is unchanged
def test_refresh_all_by_id(db: Session) -> None:
    obj_in_1 = TestCreate()
    obj_in_2 = TestCreate()
    obj_in_3 = TestCreate()
    obj = crud.crud_test.create(db=db, obj_in=obj_in_1)
    obj2 = crud.crud_test.create(db=db, obj_in=obj_in_2)
    obj3 = crud.crud_test.create(db=db, obj_in=obj_in_3)

    # assert obj and obj2 are empty now that obj3 is created, happens D2 sqlalchemy expiration
    assert not jsonable_encoder(obj)
    assert not jsonable_encoder(obj2)

    # update obj2 and 3
    new_bool_field = True
    obj2_update = TestUpdate(bool_field=new_bool_field)
    obj3_update = TestUpdate(bool_field=new_bool_field)

    # update and assert that fields were refreshed appropriately
    crud.crud_test.update(db=db, db_obj=obj2, obj_in=obj2_update)
    assert 'bool_field' in jsonable_encoder(obj2)

    # update and assert that fields were refreshed appropriately but obj2 expired
    crud.crud_test.update(db=db, db_obj=obj3, obj_in=obj3_update)
    assert not jsonable_encoder(obj2)
    assert 'bool_field' in jsonable_encoder(obj3)

    # refresh all objects
    objs = crud.crud_test.refresh_all_by_id(db=db, db_obj_ids=[obj.id, obj2.id, obj3.id])

    # assert objs are same id with bool_field updated, and other fields unchanged
    assert obj.id == objs[0].id
    assert obj2.id == objs[1].id
    assert obj3.id == objs[2].id
    assert objs[0].bool_field is False
    assert objs[1].bool_field == new_bool_field
    assert objs[2].bool_field == new_bool_field
    assert objs[0].title == 'title'
    assert objs[1].title == 'title'
    assert objs[2].title == 'title'


# given three objects to create, create them and assert they are not None and have the correct custom fields
def test_create_all_using_id(db: Session) -> None:
    obj_in_list = [TestCreate(title="1st"), TestCreate(title="2nd"), TestCreate(title="3rd")]
    objs = crud.crud_test.create_all_using_id(db=db, obj_in_list=obj_in_list)

    # assert objs are not None
    assert objs[0] is not None and objs[0].title == "1st"
    assert objs[1] is not None and objs[1].title == "2nd"
    assert objs[2] is not None and objs[2].title == "3rd"
