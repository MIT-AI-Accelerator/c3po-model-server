from app.dependencies import get_lstm_stress_classifier_model
from app.aimodels.lstm_stress_classifier.ai_service.inference.inference_model import LstmStressClassifierModel


# ensure get_lstm_stress_classifier_model returns a model
def test_get_lstm_stress_classifier_model_returns_model():
    assert isinstance(get_lstm_stress_classifier_model(), LstmStressClassifierModel)
