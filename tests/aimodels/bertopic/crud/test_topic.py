from sqlalchemy.orm import Session
from app.aimodels.bertopic.crud.crud_topic import topic_summary


def test_get_bertopic_summary_no_id(db: Session):
    assert topic_summary.get_by_model_id(db, model_id='') is None
