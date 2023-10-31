from unittest.mock import create_autospec
from pydantic import ValidationError
import pytest
import numpy as np
from bertopic import BERTopic
from plotly.graph_objs import Figure

from app.aimodels.bertopic.models.bertopic_embedding_pretrained import (
    BertopicEmbeddingPretrainedModel,
)
from app.aimodels.bertopic.ai_services.basic_inference import (
    BasicInferenceOutputs,
    BuildTopicModelInputs,
    CalculateDocumentEmbeddingsInputs,
    InitInputs,
)


def test_embedding_pretrained_model_obj_type_and_uploaded():
    with pytest.raises(ValidationError) as excinfo:
        InitInputs(
            embedding_pretrained_model_obj=BertopicEmbeddingPretrainedModel(
                model_type=None, uploaded=False
            ),
            s3=None,
        )
    assert "embedding_pretrained_model_obj must have model_type" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        InitInputs(
            embedding_pretrained_model_obj=BertopicEmbeddingPretrainedModel(
                model_type="bert", uploaded=False
            ),
            s3=None,
        )
    assert "embedding_pretrained_model_obj must be uploaded" in str(excinfo.value)


def test_basic_inference_outputs_embeddings_validator():
    with pytest.raises(ValidationError) as excinfo:
        BasicInferenceOutputs(
            documents=[],
            embeddings=[[1.0, 2.0, 3.0]],
            updated_document_indicies=[],
            topic_model=create_autospec(BERTopic),
            topics=[],
            model_word_visualization = Figure(),
            model_cluster_visualization = Figure(),
            model_timeline_visualization = Figure(),
            topic_timeline_visualization = []
        )
    assert "embeddings must be same length as documents" in str(excinfo.value)


def test_basic_inference_outputs_updated_document_indicies_validator():
    with pytest.raises(ValidationError) as excinfo:
        BasicInferenceOutputs(
            documents=[],
            embeddings=[],
            updated_document_indicies=[True],
            topic_model=create_autospec(BERTopic),
            topics=[],
            topic_word_visualization="",
            topic_cluster_visualization="",
        )
    assert "updated_document_indicies must be same length as documents" in str(
        excinfo.value
    )


def test_calculate_document_embeddings_inputs_documents_text_list_validator():
    with pytest.raises(ValidationError) as excinfo:
        CalculateDocumentEmbeddingsInputs(
            documents_text_list=[], precalculated_embeddings=None
        )
    assert "documents_text_list must be non-empty" in str(excinfo.value)


def test_calculate_document_embeddings_inputs_precalculated_embeddings_validator():
    with pytest.raises(ValidationError) as excinfo:
        CalculateDocumentEmbeddingsInputs(
            documents_text_list=["test", "test2"],
            precalculated_embeddings=[[1.0, 2.0, 3.0]],
        )
    assert (
        "precalculated_embeddings must be same length as documents_text_list if given"
        in str(excinfo.value)
    )


def test_build_topic_model_inputs_documents_text_list_validator():
    with pytest.raises(ValidationError) as excinfo:
        BuildTopicModelInputs(
            documents_text_list=[],
            document_timestamps=[],
            embeddings=np.array([[1.0, 2.0, 3.0]]),
            num_topics=2,
            seed_topic_list=[],
        )
    assert "documents_text_list must have at least 7 documents" in str(excinfo.value)


def test_build_topic_model_inputs_embeddings_validator():
    with pytest.raises(ValidationError) as excinfo:
        BuildTopicModelInputs(
            documents_text_list=["test"] * 7,
            document_timestamps=[],
            embeddings=np.array([[1.0, 2.0, 3.0]] * 6),
            num_topics=2,
            seed_topic_list=[[]],
        )
    assert "embeddings must be same length as documents_text_list" in str(excinfo.value)


def test_build_topic_model_inputs_num_topics_validator():
    with pytest.raises(ValidationError) as excinfo:
        BuildTopicModelInputs(
            documents_text_list=["test"] * 7,
            document_timestamps=[],
            embeddings=np.array([[1, 2, 3]] * 7),
            num_topics=1,
            seed_topic_list=[],
        )
    assert "num_topics must be at least 2" in str(excinfo.value)
