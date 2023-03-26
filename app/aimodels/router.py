from fastapi import APIRouter
from .lstm_stress_classifier.prefix_router import router as lstm_stress_classifier_router

router = APIRouter(
    prefix="/aimodels"
)

router.include_router(lstm_stress_classifier_router)
