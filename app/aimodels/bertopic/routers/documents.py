import os
from typing import Union
from fastapi import Depends, APIRouter
from pydantic import UUID4
from ..schemas.document import Document, DocumentCreate
from app.dependencies import get_db
from sqlalchemy.orm import Session
from .. import crud
from ..models.document import DocumentModel
from app.core.errors import HTTPValidationError, ValidationError

router = APIRouter(
    prefix="/documents"
)

@router.post(
    "/",
    response_model=Union[list[Document], HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Create documents from list",
    response_description="List of created document objects"
)
def create_document_objects_post(documents: list[DocumentCreate], db: Session = Depends(get_db)) -> (
    Union[list[Document], HTTPValidationError]
):
    """
    List of created document objects.
    """

    # pydantic handles validation
    return crud.document.create_all_using_id(db, obj_in_list=documents)
