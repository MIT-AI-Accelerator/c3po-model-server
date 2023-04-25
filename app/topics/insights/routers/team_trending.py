from pydantic import BaseModel
from fastapi import APIRouter
# from app.aimodels.bertopic.ai_service.inference import load_model

router = APIRouter(
    prefix="/teamtrending"
)


class VisualizationRequest(BaseModel):
    team: str


class VisualizationResponse(BaseModel):
    plot_params: dict


@router.post("/visualization/", response_model=VisualizationResponse)
def predict(request: VisualizationRequest):
    return VisualizationResponse(
        plot_params={}
    )
