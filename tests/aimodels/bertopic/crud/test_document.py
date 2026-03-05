from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.aimodels.bertopic.crud.crud_document import document
from app.aimodels.bertopic.models.document import DocumentModel

# assert document was built with correct model
def test_document():
    assert document.model == DocumentModel

def test_get_by_created_date_range(db: Session):

    # Create mock data with offset-naive datetimes
    end_date = datetime.now().replace(tzinfo=None)
    mock_documents = [
        DocumentModel(original_created_time=end_date-timedelta(seconds=i))
        for i in range(10)
    ]
    db.add_all(mock_documents)
    db.commit()

    # Define custom date range
    start_date = end_date - timedelta(seconds=7)
    end_date = end_date - timedelta(seconds=2)

    # Call the method with custom date range
    results = document.get_by_created_date_range(
        db, start_date=start_date, end_date=end_date
    )

    # Assert that results fall within the specified range
    assert len(results) == 6
    assert all(
        start_date <= doc.original_created_time.replace(tzinfo=None) <= end_date for doc in results
    )

    # Call the method with a limit of 2
    results = document.get_by_created_date_range(db, start_date=start_date, end_date=end_date, limit=2)

    # Assert that the number of results is limited to 2
    assert len(results) == 2
