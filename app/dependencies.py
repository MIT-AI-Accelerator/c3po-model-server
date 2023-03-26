from .aimodels.lstm_stress_classifier.ai_service.inference.inference_model import LstmStressClassifierModel

# use for DI
lstm_stress_classifier_model = LstmStressClassifierModel()
def get_lstm_stress_classifier_model():
    return lstm_stress_classifier_model
