# this is a basic crud file for "Test" entity
from app.crud.base import CRUDBase
from .crud_test_model import EmptyTestModel
from .crud_test_schema import EmptyTestCreate, EmptyTestUpdate

# CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType])
class CRUDTest(CRUDBase[EmptyTestModel, EmptyTestCreate, EmptyTestUpdate]):
    pass

crud_test = CRUDTest(EmptyTestModel)
