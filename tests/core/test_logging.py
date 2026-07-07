import logging
from unittest import mock
from app.core.logging import LogConfig, SuppressSpecificLogItemFilter, logger


def test_log_config_defaults():
    """Test that LogConfig has correct default values"""
    config = LogConfig()
    assert config.LOGGER_NAME == "transformers"
    assert config.LOG_FORMAT == "%(levelprefix)s | %(asctime)s | %(message)s"
    assert config.version == 1
    assert config.disable_existing_loggers is False


def test_log_config_formatters():
    """Test that LogConfig has formatters configured"""
    config = LogConfig()
    assert "default" in config.formatters
    assert config.formatters["default"]["fmt"] == config.LOG_FORMAT
    assert config.formatters["default"]["datefmt"] == "%Y-%m-%d %H:%M:%S"


def test_log_config_handlers():
    """Test that LogConfig has handlers configured"""
    config = LogConfig()
    assert "default" in config.handlers
    assert config.handlers["default"]["formatter"] == "default"
    assert config.handlers["default"]["class"] == "logging.StreamHandler"


def test_log_config_loggers():
    """Test that LogConfig has loggers configured"""
    config = LogConfig()
    assert "transformers" in config.loggers
    assert config.loggers["transformers"]["handlers"] == ["default"]


def test_suppress_specific_log_item_filter_init():
    """Test that SuppressSpecificLogItemFilter can be initialized"""
    filter_obj = SuppressSpecificLogItemFilter(filter_string="test_string")
    assert filter_obj.filter_string == "test_string"


def test_suppress_specific_log_item_filter_init_default():
    """Test that SuppressSpecificLogItemFilter has default empty string"""
    filter_obj = SuppressSpecificLogItemFilter()
    assert filter_obj.filter_string == ""


def test_suppress_specific_log_item_filter_blocks_matching_message():
    """Test that filter blocks messages containing the filter string"""
    filter_obj = SuppressSpecificLogItemFilter(filter_string="sensitive_data")
    
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="This contains sensitive_data in the message",
        args=(),
        exc_info=None
    )
    
    assert filter_obj.filter(record) is False


def test_suppress_specific_log_item_filter_allows_non_matching_message():
    """Test that filter allows messages not containing the filter string"""
    filter_obj = SuppressSpecificLogItemFilter(filter_string="sensitive_data")
    
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="This is a normal log message",
        args=(),
        exc_info=None
    )
    
    assert filter_obj.filter(record) is True


def test_suppress_specific_log_item_filter_empty_string():
    """Test that filter with empty string blocks all messages"""
    filter_obj = SuppressSpecificLogItemFilter(filter_string="")
    
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Any message",
        args=(),
        exc_info=None
    )
    
    assert filter_obj.filter(record) is False


def test_suppress_specific_log_item_filter_case_sensitive():
    """Test that filter is case sensitive"""
    filter_obj = SuppressSpecificLogItemFilter(filter_string="ERROR")
    
    record_upper = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="This has ERROR in it",
        args=(),
        exc_info=None
    )
    assert filter_obj.filter(record_upper) is False
    
    record_lower = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="This has error in it",
        args=(),
        exc_info=None
    )
    assert filter_obj.filter(record_lower) is True


def test_logger_exists():
    """Test that logger is properly initialized"""
    assert logger is not None
    assert logger.name == "transformers"


def test_logger_has_filter():
    """Test that logger has the SuppressSpecificLogItemFilter attached"""
    filters = logger.filters
    assert len(filters) > 0
    
    has_suppress_filter = any(isinstance(f, SuppressSpecificLogItemFilter) for f in filters)
    assert has_suppress_filter


def test_logger_filter_string():
    """Test that logger has the correct filter string configured"""
    filters = [f for f in logger.filters if isinstance(f, SuppressSpecificLogItemFilter)]
    assert len(filters) > 0
    assert filters[0].filter_string == "this_should_be_filtered_out"


def test_log_config_model_validation():
    """Test that LogConfig is a valid Pydantic model"""
    config = LogConfig()
    config_dict = config.model_dump()
    assert isinstance(config_dict, dict)
    assert "LOGGER_NAME" in config_dict
    assert "LOG_FORMAT" in config_dict
    assert "LOG_LEVEL" in config_dict
