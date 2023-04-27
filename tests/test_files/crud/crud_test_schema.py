# this is a basic schema file for "Test" entity
from typing import Optional

from pydantic import BaseModel, UUID4


# Shared properties
class TestBase(BaseModel):
    bool_field: Optional[bool] = None
    title: Optional[str] = None

# Properties to receive on Test creation
class TestCreate(TestBase):
    pass

# Properties to receive on Test update
class TestUpdate(TestBase):
    pass

# Properties shared by models stored in DB
class TestInDBBase(TestBase):
    id: UUID4
    bool_field: bool
    title: str

    class Config:
        orm_mode = True

# Properties to return to client
class Test(TestInDBBase):
    pass

# Properties properties stored in DB
class TestInDB(TestInDBBase):
    pass
