from pydantic import BaseModel
from fastapi import Depends, APIRouter

from app.dependencies import get_lstm_stress_classifier_model
from app.aimodels.lstm_stress_classifier.ai_service.inference.inference_model import LstmStressClassifierModel

router = APIRouter(
    prefix=""
)

class PromptRequest(BaseModel):
    text: str

class PromptResponse(BaseModel):
    answer: str

@router.post("/getchatstress/", response_model=PromptResponse)
def predict(request: PromptRequest, model: LstmStressClassifierModel = Depends(get_lstm_stress_classifier_model)):
    answer = model.classify_single_label(request.text)

    # convert to low / medium / high labels
    response_answer = answer
    if answer == "review":
        response_answer = "medium"
    elif answer == "action":
        response_answer = "high"
    elif answer == "recycle":
        response_answer = "low"

    return PromptResponse(
        answer=response_answer
    )
