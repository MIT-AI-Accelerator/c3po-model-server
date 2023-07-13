import pytest

@pytest.fixture(scope="function")
def test_file_dir() -> str:
    return "./tests/test_files/"
