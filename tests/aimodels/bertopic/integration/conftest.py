import pytest
from app.main import app
from app.aimodels.bertopic.routers.bertopic_embedding_pretrained import get_db

def setup(db):
    def replace_db():
        return db

    app.dependency_overrides = {get_db: replace_db}


def teardown():
    app.dependency_overrides = {}

@pytest.fixture(scope="function", autouse=True)
def setup_teardown(db):
    setup(db)
    yield
    teardown()
