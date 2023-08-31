from app.crud.base import CRUDBase
from ..models.document import DocumentModel
from ..schemas.document import DocumentCreate
from sqlalchemy.orm import Session
from typing import List
import datetime


class CRUDDocument(CRUDBase[DocumentModel, DocumentCreate, DocumentCreate]):
    def get_by_created_date_range(
        self, db: Session, *, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> List[DocumentModel]:

        if not end_date:
            end_date = datetime.datetime.now()

        if not start_date:
            start_date = end_date - datetime.timedelta(days=45)

        return (
            db.query(self.model)
            .filter(self.model.original_created_time.between(start_date, end_date))
            .all()
        )


document = CRUDDocument(DocumentModel)
