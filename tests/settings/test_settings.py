import os
from app.settings.settings import EnvironmentSettings, Settings, settings, get_env_file

BASEDIR = os.path.join(os.path.abspath(os.path.dirname("./app/settings/settings.py")), "env_var")

def test_settings_exists():
    assert isinstance(settings, Settings)

def test_environment_local_default():
    environment_settings = EnvironmentSettings()
    assert environment_settings.environment == 'local'

def test_env_file_name_local():
    environment_settings = EnvironmentSettings(environment='local')
    env_file = get_env_file(environment_settings)
    assert env_file == (os.path.join(BASEDIR, "local.env"), os.path.join(BASEDIR, "secrets.env"))

def test_env_file_name_staging():
    environment_settings = EnvironmentSettings(environment='staging')
    env_file = get_env_file(environment_settings)
    assert env_file == os.path.join(BASEDIR, "staging.env")

def test_env_file_name_production():
    environment_settings = EnvironmentSettings(environment='production')
    env_file = get_env_file(environment_settings)
    assert env_file == os.path.join(BASEDIR, "production.env")
