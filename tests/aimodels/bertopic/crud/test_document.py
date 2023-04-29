from app.aimodels.bertopic.crud.crud_document import document
from app.aimodels.bertopic.models.document import DocumentModel

# assert document was built with correct model
def test_document():
    assert document.model == DocumentModel
