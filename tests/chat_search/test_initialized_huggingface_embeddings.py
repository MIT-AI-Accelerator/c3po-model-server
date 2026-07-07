from unittest.mock import MagicMock, patch
import pytest
from app.chat_search.ai_services.initialized_huggingface_embeddings import InitializedHuggingFaceEmbeddings


def test_initialized_huggingface_embeddings_class_exists():
    """Test that InitializedHuggingFaceEmbeddings class exists"""
    assert InitializedHuggingFaceEmbeddings is not None


def test_initialized_huggingface_embeddings_has_loaded_model_annotation():
    """Test that the class has loaded_model as a type annotation"""
    # Check that the class defines loaded_model
    assert 'loaded_model' in InitializedHuggingFaceEmbeddings.__annotations__


def test_initialized_huggingface_embeddings_inherits_from_huggingface():
    """Test that InitializedHuggingFaceEmbeddings inherits from HuggingFaceEmbeddings"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    assert issubclass(InitializedHuggingFaceEmbeddings, HuggingFaceEmbeddings)


def test_initialized_huggingface_embeddings_init_success():
    """Test successful initialization of HuggingFace embeddings"""
    with patch('app.chat_search.ai_services.initialized_huggingface_embeddings.HuggingFaceEmbeddings.__init__', return_value=None):
        embeddings = InitializedHuggingFaceEmbeddings(model_name="test-model")
        assert embeddings is not None


def test_initialized_huggingface_embeddings_init_with_kwargs():
    """Test initialization with various kwargs"""
    with patch('app.chat_search.ai_services.initialized_huggingface_embeddings.HuggingFaceEmbeddings.__init__', return_value=None):
        embeddings = InitializedHuggingFaceEmbeddings(
            model_name="test-model",
            cache_folder="/tmp/cache",
            encode_kwargs={"normalize_embeddings": True}
        )
        assert embeddings is not None


def test_initialized_huggingface_embeddings_empty_kwargs():
    """Test initialization with no kwargs"""
    with patch('app.chat_search.ai_services.initialized_huggingface_embeddings.HuggingFaceEmbeddings.__init__', return_value=None):
        embeddings = InitializedHuggingFaceEmbeddings()
        assert embeddings is not None


def test_initialized_huggingface_embeddings_has_init_method():
    """Test that the class has custom __init__ method"""
    # Verify the class overrides __init__
    assert '__init__' in InitializedHuggingFaceEmbeddings.__dict__
    
    # The __init__ accepts kwargs
    import inspect
    sig = inspect.signature(InitializedHuggingFaceEmbeddings.__init__)
    assert 'kwargs' in sig.parameters
