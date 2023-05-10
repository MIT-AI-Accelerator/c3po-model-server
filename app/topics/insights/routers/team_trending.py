import json
import os
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
async def predict(request: VisualizationRequest):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_open.json")
    with open(path, "r", encoding='utf-8') as file:
        output = json.load(file)

        return VisualizationResponse(
            plot_params=output
        )
