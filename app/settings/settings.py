from pydantic import BaseSettings

# load the environment name, local, staging, or production
class EnvironmentSettings(BaseSettings):
    environment: str = "local"

# object to get other env vars
class Settings(BaseSettings):
    docs_ui_root_path: str = ""
    api_prefix: str = ""
    postgres_db_str: str = "test"

def get_env_file(environment_settings_in):

    # final output, settings object, is built
    env_file = ""

    if environment_settings_in.environment == 'production':
        env_file = "production.env"
    elif environment_settings_in.environment == 'staging':
        env_file = "staging.env"
    else:
        env_file = "local.env"

    return env_file


environment_settings = EnvironmentSettings()
settings = Settings(_env_file=get_env_file(
    environment_settings), _env_file_encoding='utf-8')
