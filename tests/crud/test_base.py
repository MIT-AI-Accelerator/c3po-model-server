from sqlalchemy.orm import Session
import uuid
from tests.test_files import crud
from tests.test_files.crud.crud_test_schema import TestCreate, TestUpdate

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
