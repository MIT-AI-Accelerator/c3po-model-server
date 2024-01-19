import os
from app.main import versioned_app
from fastapi.testclient import TestClient
from app.aimodels.lstm_stress_classifier.ai_service.inference.inference_model import BASE_CKPT_DIR

main_client = TestClient(versioned_app)

# completes succesfully when given a file
# TODO: delete file
def test_upload_checkpoint_metadata_success(test_file_dir: str):
    response = main_client.post("/backend/aimodels/lstmstressclassifier/upload-checkpoint-metadata/",
                           files={"new_file": ("filename", open(test_file_dir + "test_ckpt_metadata", "rb"))})

    assert response.status_code == 200
    assert os.path.isfile(os.path.join(BASE_CKPT_DIR, "checkpoint"))


# completes succesfully when given a file
# TODO: delete file
def test_upload_checkpoint_index_success(test_file_dir: str):
    response = main_client.post("/backend/aimodels/lstmstressclassifier/upload-checkpoint-index/",
                           files={"new_file": ("filename", open(test_file_dir + "test_ckpt_index", "rb"))})

    assert response.status_code == 200
    assert os.path.isfile(os.path.join(BASE_CKPT_DIR, "my_ckpt.index"))

# completes succesfully when given a file
# TODO: delete file
def test_upload_checkpoint_data_success(test_file_dir: str):
    response = main_client.post("/backend/aimodels/lstmstressclassifier/upload-checkpoint-data/",
                           files={"new_file": ("filename", open(test_file_dir + "test_ckpt_data", "rb"))})

    assert response.status_code == 200
    assert os.path.isfile(os.path.join(BASE_CKPT_DIR, "my_ckpt.data-00000-of-00001"))

# completes succesfully when given a file
# TODO: delete file
def test_upload_train_data_success(test_file_dir: str):
    response = main_client.post("/backend/aimodels/lstmstressclassifier/upload-train-data/",
                           files={"new_file": ("filename", open(test_file_dir + "example_data.csv", "rb"))})

    assert response.status_code == 200
    assert os.path.isfile(os.path.join(BASE_CKPT_DIR, "train_data.csv"))
