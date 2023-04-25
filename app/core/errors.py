from typing import Optional, Union
from pydantic import BaseModel, Field

class ValidationError(BaseModel):
    loc: list[Union[str, int]] = Field(..., title='Location')
    msg: str = Field(..., title='Message')
    type: str = Field(..., title='Error Type')

class HTTPValidationError(BaseModel):
    detail: Optional[list[ValidationError]] = Field(None, title='Detail')
