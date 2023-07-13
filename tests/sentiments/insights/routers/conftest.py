import pytest
from types import SimpleNamespace
from app.main import app
from app.sentiments.insights.routers.get_chat_stress import get_lstm_stress_classifier_model

def setup():
    def mock_get_lstm_stress_classifier_model():
        def classify_single_label(chat):
            return "recycle"

        return SimpleNamespace(classify_single_label=classify_single_label)

    app.dependency_overrides = {get_lstm_stress_classifier_model: mock_get_lstm_stress_classifier_model}

def teardown():
    app.dependency_overrides = {}

@pytest.fixture(scope="function", autouse=True)
def setup_teardown():
    setup()
    yield
    teardown()
