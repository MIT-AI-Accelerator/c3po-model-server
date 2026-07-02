import os
from unittest import mock
from app.core.config import (
    EnvironmentSettings,
    Settings,
    settings,
    get_env_file,
    get_originated_from,
    set_acronym_dictionary,
    get_acronym_dictionary,
    append_icao_dictionary,
    get_icao_dictionary,
    set_label_dictionary,
    get_label_dictionary,
    OriginationEnum
)

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

@mock.patch.dict(os.environ, {"ENVIRONMENT": "integration"})
def test_env_file_name_test():
    environment_settings = EnvironmentSettings()
    env_file = get_env_file(environment_settings)
    assert env_file == os.path.join(BASEDIR, "integration.env")

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

@mock.patch.dict(os.environ, {"S3_ENDPOINT_URL": "//test.com"})
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

# Tests for OriginationEnum
def test_origination_enum_app():
    assert OriginationEnum.ORIGINATED_FROM_APP == "app"
    assert OriginationEnum.ORIGINATED_FROM_APP.value == "app"

def test_origination_enum_test():
    assert OriginationEnum.ORIGINATED_FROM_TEST == "test"
    assert OriginationEnum.ORIGINATED_FROM_TEST.value == "test"

def test_get_originated_from():
    result = get_originated_from()
    assert isinstance(result, OriginationEnum)
    assert result in [OriginationEnum.ORIGINATED_FROM_APP, OriginationEnum.ORIGINATED_FROM_TEST]

# Tests for acronym dictionary functions
def test_set_acronym_dictionary():
    test_dict = {"NASA": "National Aeronautics and Space Administration", "FBI": "Federal Bureau of Investigation"}
    result = set_acronym_dictionary(test_dict)
    assert result == test_dict
    assert settings.acronym_dictionary == test_dict

def test_get_acronym_dictionary():
    test_dict = {"CIA": "Central Intelligence Agency"}
    set_acronym_dictionary(test_dict)
    result = get_acronym_dictionary()
    assert result == test_dict

def test_acronym_dictionary_empty():
    set_acronym_dictionary({})
    result = get_acronym_dictionary()
    assert result == {}

# Tests for ICAO dictionary functions
def test_append_icao_dictionary():
    # Reset ICAO dictionary first
    settings.icao_dictionary = {}
    test_dict = {"KJFK": "John F. Kennedy International Airport"}
    result = append_icao_dictionary(test_dict)
    assert "KJFK" in result
    assert result["KJFK"] == "John F. Kennedy International Airport"

def test_append_icao_dictionary_multiple():
    # Reset ICAO dictionary first
    settings.icao_dictionary = {}
    first_dict = {"KLAX": "Los Angeles International Airport"}
    second_dict = {"KORD": "O'Hare International Airport"}
    
    append_icao_dictionary(first_dict)
    result = append_icao_dictionary(second_dict)
    
    assert "KLAX" in result
    assert "KORD" in result
    assert len(result) == 2

def test_get_icao_dictionary():
    settings.icao_dictionary = {"KSFO": "San Francisco International Airport"}
    result = get_icao_dictionary()
    assert result == {"KSFO": "San Francisco International Airport"}

def test_icao_dictionary_update_existing():
    settings.icao_dictionary = {"KBOS": "Boston Logan International"}
    append_icao_dictionary({"KBOS": "Boston Logan International Airport"})
    result = get_icao_dictionary()
    assert result["KBOS"] == "Boston Logan International Airport"

# Tests for label dictionary functions
def test_set_label_dictionary():
    test_dict = {'labeling_terms': [['test1', 'test2'], ['test3', 'test4']]}
    result = set_label_dictionary(test_dict)
    assert result == test_dict
    assert settings.label_dictionary == test_dict

def test_get_label_dictionary():
    test_dict = {'labeling_terms': [['hello', 'goodbye']]}
    set_label_dictionary(test_dict)
    result = get_label_dictionary()
    assert result == test_dict

def test_label_dictionary_default():
    # Test that the default label dictionary exists
    default_dict = get_label_dictionary()
    assert 'labeling_terms' in default_dict
    assert isinstance(default_dict['labeling_terms'], list)

def test_label_dictionary_complex_structure():
    complex_dict = {
        'labeling_terms': [
            ['term1', 'term2', 'term3'],
            ['term4'],
            ['term5', 'term6']
        ],
        'metadata': 'test'
    }
    result = set_label_dictionary(complex_dict)
    assert result == complex_dict
    assert get_label_dictionary() == complex_dict
