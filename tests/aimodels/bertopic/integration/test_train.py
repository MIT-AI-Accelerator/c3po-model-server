from fastapi.testclient import TestClient
from app.aimodels.bertopic.models.document import DocumentModel

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from sample_data import CHAT_DATASET_4_PATH
from app.core.config import environment_settings


# test train endpoint with valid request
def test_train_valid_request(client: TestClient, db: Session):
    if environment_settings.environment == 'test':
        return

    my_model = 'all-MiniLM-L6-v2'
    response = client.get(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        params=jsonable_encoder({'model_name': my_model})
    )
    assert response.status_code == 200
    assert response.json()['model_name'] == my_model
    sentence_transformer_id = response.json()['id']

    # get documents
    n_docs = 20
    documents_db = db.query(DocumentModel).where(DocumentModel.text != None).limit(n_docs).all()
    assert len(documents_db) == n_docs

    body = {
        "sentence_transformer_id": sentence_transformer_id,
        "document_ids": [str(d.id) for d in documents_db]
    }

    response = client.post(
        "/aimodels/bertopic/model/train",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['id'] is not None



# test train endpoint with valid request
def test_train_valid_request_seed_topics(client: TestClient, db: Session):
    if environment_settings.environment == 'test':
        return

    my_model = 'all-MiniLM-L6-v2'
    response = client.get(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        params=jsonable_encoder({'model_name': my_model})
    )
    assert response.status_code == 200
    assert response.json()['model_name'] == my_model
    sentence_transformer_id = response.json()['id']

    # ensure init script run before this
    n_docs = 20
    documents_db = db.query(DocumentModel).where(DocumentModel.text != None).limit(n_docs).all()
    assert len(documents_db) == n_docs

    seed_topics = [['urgent', 'priority'],
        ['delay', 'slip']]
    num_topics=len(seed_topics)

    body = {
        "sentence_transformer_id": sentence_transformer_id,
        "document_ids": [d.id for d in documents_db],
        'num_topics': num_topics,
        'seed_topics': seed_topics
    }

    response = client.post(
        "/aimodels/bertopic/model/train",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['id'] is not None



# test train endpoint with valid request
def test_train_valid_request_weak_learning(client: TestClient, db: Session):
    if environment_settings.environment == 'test':
        return

    # get valid sentence transformer object
    my_model = 'all-MiniLM-L6-v2'
    response = client.get(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        params=jsonable_encoder({'model_name': my_model})
    )
    assert response.status_code == 200
    assert response.json()['model_name'] == my_model
    sentence_transformer_id = response.json()['id']

    # get valid weak learner object
    my_model = CHAT_DATASET_4_PATH.split('/')[-1]
    response = client.get(
        "/aimodels/bertopic/bertopic-embedding-pretrained",
        headers={},
        params=jsonable_encoder({'model_name': my_model})
    )
    assert response.status_code == 200
    assert response.json()['model_name'] == my_model
    weak_learner_id = response.json()['id']

    # get documents
    n_docs = 30
    documents_db = db.query(DocumentModel).where(DocumentModel.text != None,
                                                 DocumentModel.original_created_time != None).limit(n_docs).all()
    assert len(documents_db) == n_docs

    # train on documents
    body = {
        "sentence_transformer_id": sentence_transformer_id,
        "weak_learner_id": weak_learner_id,
        "document_ids": [d.id for d in documents_db]
    }

    response = client.post(
        "/aimodels/bertopic/model/train",
        headers={},
        json=jsonable_encoder(body),
    )

    assert response.status_code == 200
    assert response.json()['id'] is not None
