from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session
from app.db.base_class import Base


class SampleModel(Base):
    """Test model for base class testing"""
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)


class AnotherTestModel(Base):
    """Another test model with a different name"""
    id = Column(Integer, primary_key=True, index=True)
    value = Column(String)


def test_base_class_exists():
    """Test that Base class is properly defined"""
    assert Base is not None


def test_tablename_auto_generation():
    """Test that __tablename__ is automatically generated from class name"""
    assert SampleModel.__tablename__ == "samplemodel"


def test_tablename_lowercase_conversion():
    """Test that __tablename__ is converted to lowercase"""
    assert AnotherTestModel.__tablename__ == "anothertestmodel"


def test_base_class_inheritance():
    """Test that models properly inherit from Base"""
    test_instance = SampleModel()
    assert isinstance(test_instance, Base)


def test_multiple_models_different_tablenames():
    """Test that different models have different table names"""
    assert SampleModel.__tablename__ != AnotherTestModel.__tablename__
    assert SampleModel.__tablename__ == "samplemodel"
    assert AnotherTestModel.__tablename__ == "anothertestmodel"


def test_base_has_name_attribute():
    """Test that Base class has __name__ attribute"""
    assert hasattr(SampleModel, '__name__')
    assert SampleModel.__name__ == 'SampleModel'


def test_model_can_define_columns():
    """Test that models can define columns"""
    assert hasattr(SampleModel, 'id')
    assert hasattr(SampleModel, 'name')
    assert hasattr(AnotherTestModel, 'value')
