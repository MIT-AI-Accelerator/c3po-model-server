import uuid
from unittest.mock import create_autospec
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder
from plotly.graph_objs import Figure
from app.main import versioned_app
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import (
    BertopicEmbeddingPretrainedModel,
    EmbeddingModelTypeEnum,
)
from app.aimodels.bertopic.crud import document, bertopic_embedding_pretrained
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.ai_services.basic_inference import (
    BasicInference,
    BasicInferenceOutputs,
)

main_client = TestClient(versioned_app)

# test train endpoint with invalid request
def test_train_invalid_request():
    body = {
        "wrong_param": "",
    }

    response = main_client.post(
        "/backend/aimodels/bertopic/model/train",
        headers = {},
        json = jsonable_encoder(body),
    )

    assert response.status_code == 422


# note: see tests/aimodels/bertopic/integration/test_train_valid_*
def test_train_valid_input_request(db: Session, valid_sha256: str, mocker):
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
    new_docs = [DocumentModel(text = "new doc %d" % i) for i in range(20)]
    documents_db = document.create_all_using_id(db, obj_in_list = new_docs)

    # create a random sentence transformer obj that we know exists in db
    sentence_transformer_id = uuid.uuid4()
    bertopic_embedding_pretrained.remove(db, id = sentence_transformer_id)
    mock_embedding_obj = BertopicEmbeddingPretrainedModel(
        id = sentence_transformer_id,
        model_type = EmbeddingModelTypeEnum.SENTENCE_TRANSFORMERS,
        model_name = "all-MiniLM-L6-v2",
        sha256 = valid_sha256,
        uploaded = True,
    )
    db.add(mock_embedding_obj)
    db.commit()
    db.refresh(mock_embedding_obj)

    # mock the BasicInference object
    mock_basic_inference = create_autospec(BasicInference)
    mocker.patch(
        "app.aimodels.bertopic.routers.train.BasicInference.__new__",
        return_value = mock_basic_inference,
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
        return_value = True,
    )

    body = {
        "sentence_transformer_id": sentence_transformer_id,
        "document_ids": [str(d.id) for d in documents_db],
    }

    response = main_client.post(
        "/backend/aimodels/bertopic/model/train",
        headers = {},
        json = jsonable_encoder(body),
    )

    # clean up the mock_embedding_obj
    bertopic_embedding_pretrained.remove(db, id = sentence_transformer_id)

    assert response.status_code == 200
    assert response.json()["id"] is not None
