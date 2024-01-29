import datetime
from typing import List
from sqlalchemy import desc
from sqlalchemy.orm import Session
from app.crud.base import CRUDBase
from ..models.document import DocumentModel
from ..schemas.document import DocumentCreate


class CRUDDocument(CRUDBase[DocumentModel, DocumentCreate, DocumentCreate]):
    def get_by_created_date_range(
        self,
        db: Session,
        *,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        limit: int = 100000
    ) -> List[DocumentModel]:
        if not end_date:
            end_date = datetime.datetime.now()

        if not start_date:
            start_date = end_date - datetime.timedelta(days=45)

        return (
            db.query(self.model)
            .filter(self.model.original_created_time.between(start_date, end_date))
            .order_by(
                desc(self.model.original_created_time)
            )  # Order by original_created_time in descending order
            .limit(limit)  # Limit the number of returned rows to 1000
            .all()
        )


document = CRUDDocument(DocumentModel)
