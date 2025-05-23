import uuid
import json
from unittest.mock import MagicMock
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from app.core.config import OriginationEnum
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel
from app.aimodels.bertopic.crud import crud_bertopic_trained, crud_bertopic_visualization, crud_topic
from app.aimodels.bertopic.models.topic import TopicSummaryModel
from app.aimodels.bertopic.models.bertopic_visualization import BertopicVisualizationModel, BertopicVisualizationTypeEnum


# test visualize_topic_clusters endpoint with invalid request
def test_get_bertopic_visualize_topic_clusters_invalid_request(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%d/visualize_topic_clusters" % 0, headers = {}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid UUID, invalid length: expected length 32 for simple format, found 1"

    response = client.get(
        "/aimodels/bertopic/model/%d/visualize_topic_clusters/json" % 0, headers = {}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid UUID, invalid length: expected length 32 for simple format, found 1"


# test visualize_topic_clusters endpoint with invalid model id
def test_get_bertopic_visualize_topic_clusters_invalid_id(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_clusters" % str(uuid.uuid4()),
        headers = {},
    )
    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_clusters/json" % str(uuid.uuid4()),
        headers = {},
    )
    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]


# test visualize_topic_words endpoint with invalid request
def test_get_bertopic_visualize_topic_words_invalid_request(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%d/visualize_topic_words" % 0, headers = {}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid UUID, invalid length: expected length 32 for simple format, found 1"

    response = client.get(
        "/aimodels/bertopic/model/%d/visualize_topic_words/json" % 0, headers = {}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid UUID, invalid length: expected length 32 for simple format, found 1"


# test visualize_topic_words endpoint with invalid model id
def test_get_bertopic_visualize_topic_words_invalid_id(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_words" % str(uuid.uuid4()),
        headers = {},
    )
    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_words/json" % str(uuid.uuid4()),
        headers = {},
    )
    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]


# test trained topic endpoint with invalid request
def test_get_bertopic_trained_topics_invalid_request(client: TestClient):
    response = client.get("/aimodels/bertopic/model/%d/topics" % 0, headers = {})

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid UUID, invalid length: expected length 32 for simple format, found 1"


# test trained topic endpoint with invalid model id
def test_get_bertopic_trained_topics_invalid_id(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/model/%s/topics" % str(uuid.uuid4()), headers = {}
    )

    assert response.status_code == 422
    assert "BERTopic trained model not found" in response.json()["detail"]


# test visualize_topic_timeline endpoint with invalid request
def test_get_bertopic_visualize_topic_timeline_invalid_request(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/topic/%d/visualize_topic_timeline" % 0, headers = {}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid UUID, invalid length: expected length 32 for simple format, found 1"

    response = client.get(
        "/aimodels/bertopic/topic/%d/visualize_topic_timeline/json" % 0, headers = {}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid UUID, invalid length: expected length 32 for simple format, found 1"


# test visualize_topic_timeline endpoint with invalid model id
def test_get_bertopic_visualize_topic_timeline_invalid_id(client: TestClient):
    response = client.get(
        "/aimodels/bertopic/topic/%s/visualize_topic_timeline" % str(uuid.uuid4()),
        headers = {},
    )
    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]

    response = client.get(
        "/aimodels/bertopic/topic/%s/visualize_topic_timeline/json" % str(uuid.uuid4()),
        headers = {},
    )
    assert response.status_code == 422
    assert "BERTopic visualization not found" in response.json()["detail"]


# test model-level visualization endpoints with valid model id
def test_get_bertopic_model_visualizations(client: TestClient, db: Session):
    trained_model_obj = BertopicTrainedModel(
        sentence_transformer_id = uuid.uuid4()
    )
    trained_model_db_obj = crud_bertopic_trained.bertopic_trained.create(
        db, obj_in = trained_model_obj
    )

    # test visualize_topic_clusters
    visualization_obj = BertopicVisualizationModel(
        model_or_topic_id = trained_model_db_obj.id,
        visualization_type = BertopicVisualizationTypeEnum.MODEL_CLUSTERS,
        html_string = "<html>hi</html>",
        json_string = "[{'name': 'a dict'}]",
    )
    crud_bertopic_visualization.bertopic_visualization.create(
        db, obj_in = visualization_obj
    )

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_clusters"
        % trained_model_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_clusters/json"
        % trained_model_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200
    assert response.json() == visualization_obj.json_string

    # test visualize_topic_words
    visualization_obj = BertopicVisualizationModel(
        model_or_topic_id = trained_model_db_obj.id,
        visualization_type = BertopicVisualizationTypeEnum.MODEL_WORDS,
        html_string = "<html>hi</html>",
        json_string = "[{'name': 'a dict'}]",
    )
    crud_bertopic_visualization.bertopic_visualization.create(
        db, obj_in = visualization_obj
    )

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_words" % trained_model_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_words/json" % trained_model_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200
    assert response.json() == visualization_obj.json_string

    # test visualize_model_timeline
    visualization_obj = BertopicVisualizationModel(
        model_or_topic_id = trained_model_db_obj.id,
        visualization_type = BertopicVisualizationTypeEnum.MODEL_TIMELINE,
        html_string = "<html>hi</html>",
        json_string = "[{'name': 'a dict'}]",
    )
    crud_bertopic_visualization.bertopic_visualization.create(
        db, obj_in = visualization_obj
    )

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_timeline" % trained_model_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_timeline/json" % trained_model_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200
    assert response.json() == visualization_obj.json_string

    response = client.get(
        "/aimodels/bertopic/model/%s/topics" % trained_model_db_obj.id, headers = {}
    )
    assert response.status_code == 200


# test topic-level visualization endpoints with valid motopicdel id
def test_get_bertopic_topic_visualizations(client: TestClient, db: Session):
    trained_model_obj = BertopicTrainedModel(
        sentence_transformer_id = uuid.uuid4()
    )
    trained_model_db_obj = crud_bertopic_trained.bertopic_trained.create(
        db, obj_in = trained_model_obj
    )

    topic_summary_obj = TopicSummaryModel(
        model_id = trained_model_db_obj.id,
        name = "a name",
        top_n_words = "some words",
        top_n_documents = dict({"0": "a document", "1": "another document"}),
        summary = "a summary"
    )
    topic_summary_db_obj = crud_topic.topic_summary.create(
        db, obj_in = topic_summary_obj
    )

    # test visualize_topic_timeline
    visualization_obj = BertopicVisualizationModel(
        model_or_topic_id = topic_summary_db_obj.id,
        visualization_type = BertopicVisualizationTypeEnum.TOPIC_TIMELINE,
        html_string = "<html>hi</html>",
        json_string = "[{'name': 'a dict'}]",
    )
    crud_bertopic_visualization.bertopic_visualization.create(
        db, obj_in = visualization_obj
    )

    response = client.get(
        "/aimodels/bertopic/topic/%s/visualize_topic_timeline"
        % topic_summary_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200

    response = client.get(
        "/aimodels/bertopic/topic/%s/visualize_topic_timeline/json"
        % topic_summary_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200
    assert response.json() == visualization_obj.json_string

    response = client.get(
        "/aimodels/bertopic/topic/%s" % topic_summary_db_obj.id, headers = {}
    )
    assert response.status_code == 200
    assert response.json()['id'] == str(topic_summary_db_obj.id)
    assert response.json()['name'] == topic_summary_db_obj.name
    assert response.json()['summary'] == topic_summary_db_obj.summary


# test trained module summaries endpoint
def test_get_bertopic_trained_models(client: TestClient, db: Session, mocker: MagicMock):

    limit = 1
    mocked_model = crud_bertopic_trained.bertopic_trained.get_trained_models(db,
                                                                             row_limit = limit,
                                                                             originated_from = OriginationEnum.ORIGINATED_FROM_TEST)
    mocker.patch(
        "app.aimodels.bertopic.crud.crud_bertopic_trained.bertopic_trained.get_trained_models",
        return_value = mocked_model
    )

    response = client.get("/aimodels/bertopic/models", headers = {}, params = {'limit': limit})
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
    response = client.get("/aimodels/bertopic/topic/%d" % 0, headers = {})

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a valid UUID, invalid length: expected length 32 for simple format, found 1"


# test topic summarization endpoint with invalid model id
def test_get_bertopic_summarization_invalid_id(client: TestClient):
    response = client.get("/aimodels/bertopic/topic/%s" % str(uuid.uuid4()), headers = {})

    assert response.status_code == 422
    assert "BERTopic topic summary not found" in response.json()["detail"]


# test visualization endpoints with valid model id
def test_get_bertopic_summary(client: TestClient, db: Session):
    topic_obj = TopicSummaryModel(
        topic_id = 0,
        name = "my topic",
        top_n_words = "some_words",
        top_n_documents = dict({"0": "a document", "1": "another document"}),
        summary = "a summary"
    )

    topic_db_obj = crud_topic.topic_summary.create(db, obj_in = topic_obj)

    response = client.get("/aimodels/bertopic/topic/%s" % topic_db_obj.id, headers = {})
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
        db, obj_in = visualization_obj
    )

    response = client.get(
        "/aimodels/bertopic/topic/%s/visualize_topic_timeline"
        % topic_db_obj.id,
        headers = {},
    )
    assert response.status_code == 200


# test trained module summaries endpoint
def test_get_bertopic_trained_models(client: TestClient, db: Session):

    limit = 0
    db_objs = crud_bertopic_trained.bertopic_trained.get_trained_models(db, row_limit = limit)
    assert len(db_objs) == limit

    response = client.get("/aimodels/bertopic/models", headers = {}, params = {'limit': limit})
    assert response.status_code == 200

    rdata = response.json()
    assert len(rdata) == limit
