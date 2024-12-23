# this is a basic schema file for "EmptyTest" entity
from typing import Optional
from pydantic import BaseModel, UUID4, ConfigDict


# Shared properties
class EmptyTestBase(BaseModel):
    bool_field: Optional[bool] = None
    title: Optional[str] = None

# Properties to receive on EmptyTest creation
class EmptyTestCreate(EmptyTestBase):
    pass

# Properties to receive on EmptyTest update
class EmptyTestUpdate(EmptyTestBase):
    pass

# Properties shared by models stored in DB
class EmptyTestInDBBase(EmptyTestBase):
    id: UUID4
    bool_field: bool
    title: str

    model_config = ConfigDict(from_attributes=True)

# Properties to return to client
class EmptyTest(EmptyTestInDBBase):
    pass

# Properties properties stored in DB
class EmptyTestInDB(EmptyTestInDBBase):
    pass
