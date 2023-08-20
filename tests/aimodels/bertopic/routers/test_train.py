import uuid
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel
from app.aimodels.bertopic.crud import crud_bertopic_trained
from app.aimodels.bertopic.models.topic import TopicSummaryModel
from app.aimodels.bertopic.crud import crud_topic


# test train endpoint with invalid request
def test_train_invalid_request(client: TestClient):
    body = {
        "wrong_param": '',
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
        "/aimodels/bertopic/model/%d/visualize_topic_clusters" % 0,
        headers={}
    )

    assert response.status_code == 422
    assert 'value is not a valid uuid' in response.json()['detail'][0]['msg']

# test visualize_topic_clusters endpoint with invalid model id
def test_get_bertopic_visualize_topic_clusters_invalid_id(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_clusters" % str(
            uuid.uuid4()),
        headers={}
    )

    assert response.status_code == 422
    assert 'BERTopic trained model not found' in response.json()['detail']

# test visualize_topic_words endpoint with invalid request
def test_get_bertopic_visualize_topic_words_invalid_request(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/model/%d/visualize_topic_words" % 0,
        headers={}
    )

    assert response.status_code == 422
    assert 'value is not a valid uuid' in response.json()['detail'][0]['msg']

# test visualize_topic_words endpoint with invalid model id
def test_get_bertopic_visualize_topic_words_invalid_id(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_words" % str(
            uuid.uuid4()),
        headers={}
    )

    assert response.status_code == 422
    assert 'BERTopic trained model not found' in response.json()['detail']

# test trained topic endpoint with invalid request
def test_get_bertopic_trained_topics_invalid_request(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/model/%d/topics" % 0,
        headers={}
    )

    assert response.status_code == 422
    assert 'value is not a valid uuid' in response.json()['detail'][0]['msg']

# test trained topic endpoint with invalid model id
def test_get_bertopic_trained_topics_invalid_id(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/model/%s/topics" % str(
            uuid.uuid4()),
        headers={}
    )

    assert response.status_code == 422
    assert 'BERTopic trained model not found' in response.json()['detail']

# test visualize_topic_timeline endpoint with invalid request
def test_get_bertopic_visualize_topic_timeline_invalid_request(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/topic/%d/visualize_topic_timeline" % 0,
        headers={}
    )

    assert response.status_code == 422
    assert 'value is not a valid uuid' in response.json()['detail'][0]['msg']

# test visualize_topic_timeline endpoint with invalid model id
def test_get_bertopic_visualize_topic_timeline_invalid_id(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/topic/%s/visualize_topic_timeline" % str(
            uuid.uuid4()),
        headers={}
    )

    assert response.status_code == 422
    assert 'BERTopic topic summary not found' in response.json()['detail']

# test visualization endpoints with valid model id
def test_get_bertopic_visualizations(client: TestClient, db: Session):

    trained_model_obj = BertopicTrainedModel(
        sentence_transformer_id=uuid.uuid4(),
        topic_cluster_visualization='<html>hi</html>',
        topic_word_visualization='<html>bye</html>'
    )

    trained_model_db_obj = crud_bertopic_trained.bertopic_trained.create(
        db, obj_in=trained_model_obj)

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_clusters" % trained_model_db_obj.id,
        headers={}
    )
    assert response.status_code == 200

    response = client.get(
        "/aimodels/bertopic/model/%s/visualize_topic_words" % trained_model_db_obj.id,
        headers={}
    )
    assert response.status_code == 200

    response = client.get(
        "/aimodels/bertopic/model/%s/topics" % trained_model_db_obj.id,
        headers={}
    )
    assert response.status_code == 200

# test topic summarization endpoint with invalid request
def test_get_bertopic_summarization_invalid_request(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/topic/%d" % 0,
        headers={}
    )

    assert response.status_code == 422
    assert 'value is not a valid uuid' in response.json()['detail'][0]['msg']

# test topic summarization endpoint with invalid model id
def test_get_bertopic_summarization_invalid_id(client: TestClient):

    response = client.get(
        "/aimodels/bertopic/topic/%s" % str(
            uuid.uuid4()),
        headers={}
    )

    assert response.status_code == 422
    assert 'BERTopic topic summary not found' in response.json()['detail']

# test visualization endpoints with valid model id
def test_get_bertopic_summary(client: TestClient, db: Session):

    topic_obj = TopicSummaryModel(
        topic_id=0,
        name='my topic',
        top_n_words='some_words',
        top_n_documents=dict({'0': 'a document', '1': 'another document'}),
        summary='a summary',
        topic_timeline_visualization='<html>it works</html>'
    )

    topic_db_obj = crud_topic.topic_summary.create(
        db, obj_in=topic_obj)

    response = client.get(
        "/aimodels/bertopic/topic/%s" % topic_db_obj.id,
        headers={}
    )
    rdata = response.json()

    assert response.status_code == 200
    assert rdata['topic_id'] == topic_obj.topic_id
    assert rdata['name'] == topic_obj.name
    assert rdata['top_n_words'] == topic_obj.top_n_words
    assert rdata['top_n_documents'] == topic_obj.top_n_documents
    assert rdata['summary'] == topic_obj.summary
    assert rdata['topic_timeline_visualization'] == topic_obj.topic_timeline_visualization
