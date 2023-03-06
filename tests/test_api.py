from types import SimpleNamespace
from fastapi.testclient import TestClient

from app.api import app, get_model

client = TestClient(app)

# ************Mocks*******************
# creates a mock empty object that is structured like the output
# of get_model for the lstm, at least regarding classify_single_label


def mock_get_model():

    def classify_single_label(chat):
        return "recycle"

    return SimpleNamespace(classify_single_label=classify_single_label)


# overrides the dependency for the tests
app.dependency_overrides[get_model] = mock_get_model
# *************************************

#***********Module Test Vars***********
test_file_dir = "./tests/test_files/"
#**************************************

# Given: A line of chat--"Hello there!" and the above mocked model that classifies it as "recycle"
# When: This line is sent to the endpoint /predict
# Then: we expect to receive a 200 and the appropriately formatted response in the body
def test_predict():
    response = client.post(
        "/predict/",
        headers={},
        json={"text": "Hello there!"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "recycle"}

# completes succesfully when given a file
# TODO: assert file exists then delete it
def test_upload_checkpoint_metadata_success():
    response = client.post("/lstm-basic-classifier/upload-checkpoint-metadata/",
                           files={"new_file": ("filename", open(test_file_dir + "test_ckpt_metadata", "rb"))})

    assert response.status_code == 200

# completes succesfully when given a file
# TODO: assert file exists then delete it
def test_upload_checkpoint_index_success():
    response = client.post("/lstm-basic-classifier/upload-checkpoint-index/",
                           files={"new_file": ("filename", open(test_file_dir + "test_ckpt_index", "rb"))})

    assert response.status_code == 200

# completes succesfully when given a file
# TODO: assert file exists then delete it
def test_upload_checkpoint_data_success():
    response = client.post("/lstm-basic-classifier/upload-checkpoint-data/",
                           files={"new_file": ("filename", open(test_file_dir + "test_ckpt_data", "rb"))})

    assert response.status_code == 200
