import os
from unittest import mock
from app.core.config import EnvironmentSettings, Settings, settings, get_env_file

BASEDIR = os.path.join(os.path.abspath(os.path.dirname("./app/core/config.py")), "env_var")

def test_settings_exists():
    assert isinstance(settings, Settings)

@mock.patch.dict(os.environ, clear=True) # clear=True is needed to clear the environment variables
def test_environment_test_default():
    environment_settings = EnvironmentSettings()
    assert environment_settings.environment == 'test'

@mock.patch.dict(os.environ, {"ENVIRONMENT": "test"})
def test_env_file_name_test():
    environment_settings = EnvironmentSettings()
    env_file = get_env_file(environment_settings)
    assert env_file == os.path.join(BASEDIR, "test.env")

@mock.patch.dict(os.environ, {"ENVIRONMENT": "local"})
def test_env_file_name_local():
    environment_settings = EnvironmentSettings()
    env_file = get_env_file(environment_settings)
    assert env_file == (os.path.join(BASEDIR, "local.env"), os.path.join(BASEDIR, "secrets.env"))


@mock.patch.dict(os.environ, {"ENVIRONMENT": "development"})
def test_env_file_name_development():
    environment_settings = EnvironmentSettings()
    env_file = get_env_file(environment_settings)
    assert env_file == (os.path.join(BASEDIR, "development.env"), os.path.join(BASEDIR, "secrets.env"))

@mock.patch.dict(os.environ, {"ENVIRONMENT": "staging"})
def test_env_file_name_staging():
    environment_settings = EnvironmentSettings()
    env_file = get_env_file(environment_settings)
    assert env_file == os.path.join(BASEDIR, "staging.env")

@mock.patch.dict(os.environ, {"ENVIRONMENT": "production"})
def test_env_file_name_production():
    environment_settings = EnvironmentSettings()
    env_file = get_env_file(environment_settings)
    assert env_file == os.path.join(BASEDIR, "production.env")

@mock.patch.dict(os.environ, {"MINIO_ENDPOINT_URL": "http://test.com"})
def test_remove_http_or_https_removes_http():
    environment_settings = EnvironmentSettings()
    mock_settings = Settings(_env_file=get_env_file(
        environment_settings), _env_file_encoding='utf-8')
    assert mock_settings.s3_endpoint_url == "test.com"

@mock.patch.dict(os.environ, {"MINIO_ENDPOINT_URL": "https://test.com"})
def test_remove_http_or_https_removes_https():
    environment_settings = EnvironmentSettings()
    mock_settings = Settings(_env_file=get_env_file(
        environment_settings), _env_file_encoding='utf-8')
    assert mock_settings.s3_endpoint_url == "test.com"

@mock.patch.dict(os.environ, {"MINIO_ENDPOINT_URL": "//test.com"})
def test_remove_http_or_https_does_nothing_if_no_http_or_https():
    environment_settings = EnvironmentSettings()
    mock_settings = Settings(_env_file=get_env_file(
        environment_settings), _env_file_encoding='utf-8')
    assert mock_settings.s3_endpoint_url == "//test.com"

@mock.patch.dict(os.environ, {"ENVIRONMENT": "test"})
def test_assemble_db_with_uri():
    environment_settings = EnvironmentSettings()
    mock_settings = Settings(_env_file=get_env_file(
        environment_settings), _env_file_encoding='utf-8')
    assert 'postgres:5432/postgres' in mock_settings.sqlalchemy_database_uri.unicode_string()
