# this is a basic crud file for "Test" entity
from app.crud.base import CRUDBase
from .crud_test_model import TestModel
from .crud_test_schema import TestCreate, TestUpdate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDTest(CRUDBase[TestModel, TestCreate, TestUpdate]):
    pass

crud_test = CRUDTest(TestModel)
