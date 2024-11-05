from collections.abc import Generator
import random
from unittest.mock import MagicMock
from .test_files.db.db_test_session import SessionLocal
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.minio import build_client
from app.core.config import OriginationEnum


# in case you are wondering why we use yield instead of return, check this
# https://stackoverflow.com/questions/64763770/why-we-use-yield-to-get-sessionlocal-in-fastapi-with-sqlalchemy
@pytest.fixture(scope="session")
def db() -> Generator:
    yield SessionLocal()

@pytest.fixture(scope="session")
def s3():
    # return not yield...we only want a single client, not a new one each time
    return build_client()

@pytest.fixture(scope="session")
def mock_s3():
    # return not yield...we only want a single client, not a new one each time
    return MagicMock()

@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        # initialize originated_from to test to allow for db cleanup
        response = c.get("/originated_from_test/")
        data = response.json()
        assert data == OriginationEnum.ORIGINATED_FROM_TEST
        assert response.status_code == 200
        yield c

@pytest.fixture(scope="function")
def valid_sha256() -> str:
    random.seed(None)
    output = '%032x' % random.getrandbits(256)

    # it is possible that the random number generator generates a number with slightly than 64 digits
    if len(output) < 64:
        num_ones = 64 - len(output)
        output = output + '1' * num_ones
    return output
