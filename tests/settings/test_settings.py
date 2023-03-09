from app.settings.settings import EnvironmentSettings, Settings, settings, get_env_file

def test_settings_exists():
    assert isinstance(settings, Settings)

def test_environment_local_default():
    environment_settings = EnvironmentSettings()
    assert environment_settings.environment == 'local'

def test_env_file_name_local():
    environment_settings = EnvironmentSettings()
    env_file = get_env_file(environment_settings)
    assert env_file == 'local.env'

def test_env_file_name_staging():
    environment_settings = EnvironmentSettings(environment='staging')
    env_file = get_env_file(environment_settings)
    assert env_file == 'staging.env'

def test_env_file_name_production():
    environment_settings = EnvironmentSettings(environment='production')
    env_file = get_env_file(environment_settings)
    assert env_file == 'production.env'
