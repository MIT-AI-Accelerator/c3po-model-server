import os
from pydantic import BaseSettings

# load the environment name, local, staging, or production
class EnvironmentSettings(BaseSettings):
    environment: str = "local"

# object to get other env vars
class Settings(BaseSettings):
    docs_ui_root_path: str = ""
    postgres_db_str: str = "test"
    log_level: str = "INFO"
    mm_token: str = ""
    mm_base_url: str = ""
    mm_aoc_base_url: str = ""
    mm_nitmre_base_url: str = ""

def get_env_file(environment_settings_in):
    BASEDIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "env_var")

    # final output, settings object, is built
    env_file = ""

    if environment_settings_in.environment == 'production':
        env_file = os.path.join(BASEDIR, "production.env")
    elif environment_settings_in.environment == 'staging':
        env_file = os.path.join(BASEDIR, "staging.env")
    else:
        # put local secrets into secrets.env and ensure on .gitignore, K8s injects staging and prod into env vars
        env_file = (os.path.join(BASEDIR, "local.env"), os.path.join(BASEDIR, "secrets.env"))
    return env_file

environment_settings = EnvironmentSettings()
settings = Settings(_env_file=get_env_file(
    environment_settings), _env_file_encoding='utf-8')
