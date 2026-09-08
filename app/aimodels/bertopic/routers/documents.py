from typing import Annotated, Union
from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.core.errors import HTTPValidationError
from app.ppg_common.schemas.bertopic.document import Document, DocumentCreate
from .. import crud

router = APIRouter(
    prefix="/documents"
)

@router.post(
    "/",
    responses={
        '200': {'model': list[Document]},
        '422': {'model': HTTPValidationError}
    },
    summary="Create documents from list",
    response_description="List of created document objects"
)
def create_document_objects_post(documents: list[DocumentCreate], db: Annotated[Session, Depends(get_db)]) -> (
    Union[list[Document], HTTPValidationError]
):
    """
    List of created document objects.
    """

    # pydantic handles validation
    return crud.document.create_all_using_id(db, obj_in_list=documents)
