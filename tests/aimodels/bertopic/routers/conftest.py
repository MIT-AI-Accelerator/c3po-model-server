import pytest
from app.main import app
from app.aimodels.bertopic.routers.bertopic_embedding_pretrained import get_db, get_minio

# see docs here for setup/teardown https://pytest.org/en/7.4.x/how-to/xunit_setup.html
def setup(db, mock_s3):
    def replace_db():
        return db

    def mock_get_minio():
        return mock_s3

    app.dependency_overrides = {get_db: replace_db, get_minio: mock_get_minio}


def teardown():
    app.dependency_overrides = {}

@pytest.fixture(scope="function", autouse=True)
def setup_teardown(db, mock_s3):
    setup(db, mock_s3)
    yield
    teardown()
