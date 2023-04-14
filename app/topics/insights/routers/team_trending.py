from pydantic import BaseModel
from fastapi import APIRouter
from app.aimodels.bertopic.ai_service.inference.inference_model import load_model

router = APIRouter(
    prefix="/teamtrending"
)


class VisualizationRequest(BaseModel):
    team: str


class VisualizationResponse(BaseModel):
    plot_params: dict


@router.post("/visualization/", response_model=VisualizationResponse)
def predict(request: VisualizationRequest):
    bert_model = load_model('1')
    print(request.team)
    return VisualizationResponse(
        plot_params=bert_model.trained_models[0].visualization_config
    )
