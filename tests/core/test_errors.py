from pydantic import ValidationError as PydanticValidationError
from app.core.errors import ValidationError, HTTPValidationError


def test_validation_error_model():
    """Test that ValidationError model can be instantiated"""
    error = ValidationError(
        loc=["body", "field"],
        msg="Field is required",
        type="value_error.missing"
    )
    assert error.loc == ["body", "field"]
    assert error.msg == "Field is required"
    assert error.type == "value_error.missing"


def test_validation_error_with_string_loc():
    """Test ValidationError with string location"""
    error = ValidationError(
        loc=["body"],
        msg="Invalid input",
        type="value_error"
    )
    assert error.loc == ["body"]


def test_validation_error_with_int_loc():
    """Test ValidationError with integer location"""
    error = ValidationError(
        loc=[0, "field"],
        msg="Invalid item",
        type="value_error"
    )
    assert error.loc == [0, "field"]


def test_validation_error_mixed_loc():
    """Test ValidationError with mixed string and int locations"""
    error = ValidationError(
        loc=["items", 0, "name"],
        msg="Field validation failed",
        type="type_error.str"
    )
    assert error.loc == ["items", 0, "name"]
    assert isinstance(error.loc[0], str)
    assert isinstance(error.loc[1], int)
    assert isinstance(error.loc[2], str)


def test_http_validation_error_with_details():
    """Test HTTPValidationError with validation error details"""
    validation_errors = [
        ValidationError(
            loc=["body", "email"],
            msg="Invalid email format",
            type="value_error.email"
        ),
        ValidationError(
            loc=["body", "age"],
            msg="Must be a positive integer",
            type="type_error.integer"
        )
    ]
    
    http_error = HTTPValidationError(detail=validation_errors)
    assert http_error.detail is not None
    assert len(http_error.detail) == 2
    assert http_error.detail[0].msg == "Invalid email format"
    assert http_error.detail[1].loc == ["body", "age"]


def test_http_validation_error_without_details():
    """Test HTTPValidationError without details (None)"""
    http_error = HTTPValidationError(detail=None)
    assert http_error.detail is None


def test_http_validation_error_empty_list():
    """Test HTTPValidationError with empty detail list"""
    http_error = HTTPValidationError(detail=[])
    assert http_error.detail == []


def test_validation_error_to_dict():
    """Test that ValidationError can be converted to dict"""
    error = ValidationError(
        loc=["query", "limit"],
        msg="Must be less than 100",
        type="value_error.number.not_le"
    )
    error_dict = error.model_dump()
    assert error_dict["loc"] == ["query", "limit"]
    assert error_dict["msg"] == "Must be less than 100"
    assert error_dict["type"] == "value_error.number.not_le"


def test_http_validation_error_to_dict():
    """Test that HTTPValidationError can be converted to dict"""
    validation_error = ValidationError(
        loc=["body", "username"],
        msg="Username already exists",
        type="value_error.unique"
    )
    http_error = HTTPValidationError(detail=[validation_error])
    error_dict = http_error.model_dump()
    
    assert "detail" in error_dict
    assert len(error_dict["detail"]) == 1
    assert error_dict["detail"][0]["loc"] == ["body", "username"]


def test_validation_error_field_titles():
    """Test that ValidationError has proper field titles"""
    # Check that the Field titles are set correctly
    from pydantic import BaseModel
    
    # ValidationError should have field metadata
    schema = ValidationError.model_json_schema()
    assert schema["properties"]["loc"]["title"] == "Location"
    assert schema["properties"]["msg"]["title"] == "Message"
    assert schema["properties"]["type"]["title"] == "Error Type"


def test_http_validation_error_field_title():
    """Test that HTTPValidationError has proper field title"""
    schema = HTTPValidationError.model_json_schema()
    assert schema["properties"]["detail"]["title"] == "Detail"


def test_validation_error_required_fields():
    """Test that ValidationError requires all fields"""
    try:
        # Try to create without required fields
        ValidationError()
        assert False, "Should have raised ValidationError"
    except PydanticValidationError:
        # Expected to fail
        assert True


def test_validation_error_json_serialization():
    """Test that ValidationError can be JSON serialized"""
    error = ValidationError(
        loc=["body", "data"],
        msg="Invalid data",
        type="value_error"
    )
    json_str = error.model_dump_json()
    assert "body" in json_str
    assert "data" in json_str
    assert "Invalid data" in json_str
