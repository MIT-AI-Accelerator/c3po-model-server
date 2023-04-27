from collections.abc import Generator
from .test_files.db.db_test_session import SessionLocal
import pytest
from fastapi.testclient import TestClient
from app.main import app


# in case you are wondering why we use yield instead of return, check this
# https://stackoverflow.com/questions/64763770/why-we-use-yield-to-get-sessionlocal-in-fastapi-with-sqlalchemy
@pytest.fixture(scope="session")
def db() -> Generator:
    yield SessionLocal()


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c
