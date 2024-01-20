from typing import TYPE_CHECKING
from sqlalchemy import Column, ForeignKey, Table, Enum
from app.db.base_class import Base

if TYPE_CHECKING:
    from .document import DocumentModel  # noqa: F401
    from .bertopic_trained import BertopicTrainedModel  # noqa: F401

# see here for basic many-to-many in fastapi without metadata: https://www.gormanalysis.com/blog/many-to-many-relationships-in-fastapi/
# pylint: disable=no-member
DocumentBertopicTrainedModel = Table('documentbertopictrainedmodel', Base.metadata,
    Column('document_id', ForeignKey('documentmodel.id'), primary_key=True),
    Column('bertopic_trained_model_id', ForeignKey('bertopictrainedmodel.id'), primary_key=True)
)
