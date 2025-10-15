import os
import enum
from typing import Optional, Any
from pydantic import PostgresDsn, field_validator, ValidationInfo
from pydantic_settings import BaseSettings

class OriginationEnum(str, enum.Enum):
    ORIGINATED_FROM_APP = "app"
    ORIGINATED_FROM_TEST = "test"

# load the environment name, local, test, staging, or production
class EnvironmentSettings(BaseSettings):
    environment: str = "test"

# object to get other env vars
class Settings(BaseSettings):
    # general settings
    docs_ui_root_path: str = ""
    log_level: str = "INFO"
    originated_from: OriginationEnum = OriginationEnum.ORIGINATED_FROM_APP
    acronym_dictionary: dict = {}
    icao_dictionary: dict = {}

    # s3 settings
    s3_bucket_name: str = ""
    s3_endpoint_url: str = ""
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = ""
    s3_secure: bool = True

    # postgreSQL settings
    postgres_user: str = ""
    postgres_password: str
    postgres_server: str = "db"
    postgres_port: str = "5432"
    postgres_db: str
    sqlalchemy_database_uri: Optional[PostgresDsn] = None

    @field_validator("sqlalchemy_database_uri", mode="after")
    def assemble_db_connection(cls, v: Optional[str], values: ValidationInfo) -> Any:
        # pylint: disable=no-self-argument

        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            username=values.data.get("postgres_user"),
            password=values.data.get("postgres_password"),
            host=values.data.get("postgres_server"),
            port=int(values.data.get("postgres_port")),
            path=values.data.get("postgres_db"),
        )

    # Mattermost settings
    mm_token: str = ""
    mm_base_url: str = ""
    mm_aoc_base_url: str = ""
    mm_nitmre_base_url: str = ""

    # Default model hash settings
    default_sha256_l13b_snoozy: str = ""
    default_sha256_q4_k_m: str = ""

    # Weak learner settings
    label_dictionary: dict = {'labeling_terms': [['joined the channel', 'added to the channel'],
                                                 ['hello', 'hola', 'good morning', 'good evening', 'good night'],
                                                 ['lunch', 'dinner', 'food']]}


def get_env_file(environment_settings_in):
    # get the base directory
    BASEDIR = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), "env_var")

    # final output, settings object, is built
    env_file = ""

    if environment_settings_in.environment == 'production':
        env_file = os.path.join(BASEDIR, "production.env")
    elif environment_settings_in.environment == 'staging':
        env_file = os.path.join(BASEDIR, "staging.env")
    elif environment_settings_in.environment == 'local':
        # put local secrets into secrets.env and ensure on .gitignore, K8s injects staging and prod into env vars
        env_file = (os.path.join(BASEDIR, "local.env"),
                    os.path.join(BASEDIR, "secrets.env"))
    elif environment_settings_in.environment == 'development':
        env_file = (os.path.join(BASEDIR, "development.env"),
                    os.path.join(BASEDIR, "secrets.env"))
    elif environment_settings_in.environment == 'integration':
        env_file = os.path.join(BASEDIR, "integration.env")
    else:
        env_file = os.path.join(BASEDIR, "test.env")

    return env_file

environment_settings = EnvironmentSettings()
settings = Settings(_env_file=get_env_file(
    environment_settings), _env_file_encoding='utf-8')

def get_originated_from():
    return settings.originated_from

# Produces dictionary of acronym, meaning pairs from a csv file
def set_acronym_dictionary(acronym_dictionary):
    settings.acronym_dictionary = acronym_dictionary
    return settings.acronym_dictionary

def get_acronym_dictionary():
    return settings.acronym_dictionary

# Functions to access icao dictionary
def append_icao_dictionary(icao_dictionary):
    settings.icao_dictionary.update(icao_dictionary)

    return settings.icao_dictionary

def get_icao_dictionary():
    return settings.icao_dictionary

# Functions to access weak learning label dictionary
def set_label_dictionary(label_dictionary):
    settings.label_dictionary = label_dictionary
    return settings.label_dictionary

def get_label_dictionary():
    return settings.label_dictionary
