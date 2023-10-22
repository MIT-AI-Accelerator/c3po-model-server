import uuid
import json
from unittest.mock import MagicMock, create_autospec
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder
from plotly.graph_objs import Figure
from app.core.config import OriginationEnum
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import (
    BertopicEmbeddingPretrainedModel,
    EmbeddingModelTypeEnum,
)
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel
from app.aimodels.bertopic.crud import crud_bertopic_trained, crud_bertopic_visualization, document, crud_topic, bertopic_embedding_pretrained
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.models.topic import TopicSummaryModel
from app.aimodels.bertopic.models.bertopic_visualization import BertopicVisualizationModel, BertopicVisualizationTypeEnum
from app.aimodels.bertopic.ai_services.basic_inference import (
    BasicInference,
    BasicInferenceOutputs,
)


# test train endpoint with invalid request
def test_train_invalid_request(client: TestClient):
    body = {
        "wrong_param": "",
    }

    response = client.post(
        "/aimodels/bertopic/model/train",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 422


# note: see tests/aimodels/bertopic/integration/test_train_valid_*


# test visualize_topic_clusters endpoint with invalid request
def test_get_bertopic_visualize_topic_clusters_invalid_request(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%d/visualize_topic_clusters" % 0, headers={}
    )

    assert response.status_code == 422
    assert "value is not a valid uuid" in response.json()["detail"][0]["msg"]


# test visualize_topic_clusters endpoint with invalid model id
def test_get_bertopic_visualize_topic_clusters_invalid_id(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_clusters" % str(uuid.uuid4()),
        headers={},
    )

    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]


# test visualize_topic_words endpoint with invalid request
def test_get_bertopic_visualize_topic_words_invalid_request(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%d/visualize_topic_words" % 0, headers={}
    )

    assert response.status_code == 422
    assert "value is not a valid uuid" in response.json()["detail"][0]["msg"]


# test visualize_topic_words endpoint with invalid model id
def test_get_bertopic_visualize_topic_words_invalid_id(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_words" % str(uuid.uuid4()),
        headers={},
    )

    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]


# test trained topic endpoint with invalid request
def test_get_bertopic_trained_topics_invalid_request(client: TestClient):
    response = client.get("/aimodels/bertopic/model/%d/topics" % 0, headers={})

    assert response.status_code == 422
    assert "value is not a valid uuid" in response.json()["detail"][0]["msg"]


# test trained topic endpoint with invalid model id
def test_get_bertopic_trained_topics_invalid_id(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%s/topics" % str(uuid.uuid4()), headers={}
    )

    assert response.status_code == 422
    assert "BERTopic trained model not found" in response.json()["detail"]


# test visualize_topic_timeline endpoint with invalid request
def test_get_bertopic_visualize_topic_timeline_invalid_request(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/topic/%d/visualize_topic_timeline" % 0, headers={}
    )

    assert response.status_code == 422
    assert "value is not a valid uuid" in response.json()["detail"][0]["msg"]


# test visualize_topic_timeline endpoint with invalid model id
def test_get_bertopic_visualize_topic_timeline_invalid_id(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/topic/%s/visualize_topic_timeline" % str(uuid.uuid4()),
        headers={},
    )

    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]


# test visualization endpoints with valid model id
def test_get_bertopic_visualizations(client: TestClient, db: Session):
    trained_model_obj = BertopicTrainedModel(
        sentence_transformer_id=uuid.uuid4()
    )
    trained_model_db_obj = crud_bertopic_trained.bertopic_trained.create(
        db, obj_in=trained_model_obj
    )

    # test visualize_topic_clusters
    visualization_obj = BertopicVisualizationModel(
        model_or_topic_id = trained_model_db_obj.id,
        visualization_type = BertopicVisualizationTypeEnum.MODEL_CLUSTERS,
        html_string = "<html>hi</html>",
        json_string = "[{'name': 'a dict'}]",
    )
    crud_bertopic_visualization.bertopic_visualization.create(
        db, obj_in=visualization_obj
    )

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_clusters"
        % trained_model_db_obj.id,
        headers={},
    )
    assert response.status_code == 200

    # test visualize_topic_words
    visualization_obj = BertopicVisualizationModel(
        model_or_topic_id = trained_model_db_obj.id,
        visualization_type = BertopicVisualizationTypeEnum.MODEL_WORDS,
        html_string = "<html>hi</html>",
        json_string = "[{'name': 'a dict'}]",
    )
    crud_bertopic_visualization.bertopic_visualization.create(
        db, obj_in=visualization_obj
    )

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_words" % trained_model_db_obj.id,
        headers={},
    )
    assert response.status_code == 200

    response = client.get(
        "/aimodels/bertopic/model/%s/topics" % trained_model_db_obj.id, headers={}
    )
    assert response.status_code == 200


# test trained module summaries endpoint
def test_get_bertopic_trained_models(client: TestClient, db: Session, mocker: MagicMock):

    limit = 1
    mocked_model = crud_bertopic_trained.bertopic_trained.get_trained_models(db,
                                                                             row_limit=limit,
                                                                             originated_from=OriginationEnum.ORIGINATED_FROM_TEST)
    mocker.patch(
        "app.aimodels.bertopic.crud.crud_bertopic_trained.bertopic_trained.get_trained_models",
        return_value=mocked_model
    )

    response = client.get("/aimodels/bertopic/models", headers={}, params={'limit': limit})
    assert response.status_code == 200

    rdata = response.json()
    summary_obj = json.loads(rdata)
    assert len(summary_obj) == limit

    assert str(mocked_model[0].id) == str(summary_obj[0]['id'])
    assert str(mocked_model[0].sentence_transformer_id) == str(summary_obj[0]['sentence_transformer_id'])
    assert str(mocked_model[0].weak_learner_id) == str(summary_obj[0]['weak_learner_id'])
    assert str(mocked_model[0].summarization_model_id) == str(summary_obj[0]['summarization_model_id'])


# test topic summarization endpoint with invalid request
def test_get_bertopic_summarization_invalid_request(client: TestClient):
    response = client.get("/aimodels/bertopic/topic/%d" % 0, headers={})

    assert response.status_code == 422
    assert "value is not a valid uuid" in response.json()["detail"][0]["msg"]


# test topic summarization endpoint with invalid model id
def test_get_bertopic_summarization_invalid_id(client: TestClient):
    response = client.get("/aimodels/bertopic/topic/%s" % str(uuid.uuid4()), headers={})

    assert response.status_code == 422
    assert "BERTopic topic summary not found" in response.json()["detail"]


# test visualization endpoints with valid model id
def test_get_bertopic_summary(client: TestClient, db: Session):
    topic_obj = TopicSummaryModel(
        topic_id=0,
        name="my topic",
        top_n_words="some_words",
        top_n_documents=dict({"0": "a document", "1": "another document"}),
        summary="a summary"
    )

    topic_db_obj = crud_topic.topic_summary.create(db, obj_in=topic_obj)

    response = client.get("/aimodels/bertopic/topic/%s" % topic_db_obj.id, headers={})
    summary = response.json()

    assert response.status_code == 200
    assert summary["topic_id"] == topic_obj.topic_id
    assert summary["name"] == topic_obj.name
    assert summary["top_n_words"] == topic_obj.top_n_words
    assert summary["top_n_documents"] == topic_obj.top_n_documents
    assert summary["summary"] == topic_obj.summary

    # test visualize_topic_words
    visualization_obj = BertopicVisualizationModel(
        model_or_topic_id = topic_db_obj.id,
        visualization_type = BertopicVisualizationTypeEnum.TOPIC_TIMELINE,
        html_string = "<html>hi</html>",
        json_string = "[{'name': 'a dict'}]",
    )
    crud_bertopic_visualization.bertopic_visualization.create(
        db, obj_in=visualization_obj
    )

    response = client.get(
        "/aimodels/bertopic/topic/%s/visualize_topic_timeline"
        % topic_db_obj.id,
        headers={},
    )
    assert response.status_code == 200


def test_train_valid_input_request(
    client: TestClient, db: Session, valid_sha256: str, mocker
):
    """
    Given:
    - a valid sentence transformer id in the database
    - a list of valid document ids in the database
    - a working BasicInference class instantiation
    - a working BasicInferenceOutputs class and output object for the train_bertopic_on_documents function
    - a working pickle_and_upload_object_to_minio function

    When:
    - the train endpoint is called with the valid sentence transformer id and list of document ids

    Then:
    - the endpoint should return a 200 status code
    - the endpoint should return a valid uuid for the newly created BertopicTrainedModel object
    """

    # create documents for training
    new_docs = [DocumentModel(text="new doc %d" % i) for i in range(20)]
    documents_db = document.create_all_using_id(db, obj_in_list=new_docs)

    # create a random sentence transformer obj that we know exists in db
    sentence_transformer_id = uuid.uuid4()
    bertopic_embedding_pretrained.remove(db, id=sentence_transformer_id)
    mock_embedding_obj = BertopicEmbeddingPretrainedModel(
        id=sentence_transformer_id,
        model_type=EmbeddingModelTypeEnum.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        sha256=valid_sha256,
        uploaded=True,
    )
    db.add(mock_embedding_obj)
    db.commit()
    db.refresh(mock_embedding_obj)

    # mock the BasicInference object
    mock_basic_inference = create_autospec(BasicInference)
    mocker.patch(
        "app.aimodels.bertopic.routers.train.BasicInference.__new__",
        return_value=mock_basic_inference,
    )

    # mock the return value of inference
    mock_inference_outputs = create_autospec(BasicInferenceOutputs)
    mock_inference_outputs.model_word_visualization = Figure()
    mock_inference_outputs.model_cluster_visualization = Figure()
    mock_inference_outputs.model_timeline_visualization = Figure()
    mock_inference_outputs.topic_timeline_visualization = []
    mock_inference_outputs.topic_model = object()
    mock_inference_outputs.topics = list()
    mock_inference_outputs.embeddings = [[1, 1] for _ in documents_db]
    mock_inference_outputs.updated_document_indicies = [True for _ in documents_db]

    # add the mocked return value to the mocked inference object
    mock_basic_inference.train_bertopic_on_documents.return_value = (
        mock_inference_outputs
    )

    # mock minio upload
    mocker.patch(
        "app.aimodels.bertopic.routers.train.pickle_and_upload_object_to_minio",
        return_value=True,
    )

    body = {
        "sentence_transformer_id": sentence_transformer_id,
        "document_ids": [str(d.id) for d in documents_db],
    }

    response = client.post(
        "/aimodels/bertopic/model/train",
        headers={},
        json=jsonable_encoder(body),
    )

    # clean up the mock_embedding_obj
    bertopic_embedding_pretrained.remove(db, id=sentence_transformer_id)

    assert response.status_code == 200
    assert response.json()["id"] is not None


# test trained module summaries endpoint
def test_get_bertopic_trained_models(client: TestClient, db: Session):

    limit = 0
    db_objs = crud_bertopic_trained.bertopic_trained.get_trained_models(db, row_limit=limit)
    assert len(db_objs) == limit

    response = client.get("/aimodels/bertopic/models", headers={}, params={'limit': limit})
    assert response.status_code == 200

    rdata = response.json()
    summary_obj = json.loads(rdata)
    assert len(summary_obj) == limit
