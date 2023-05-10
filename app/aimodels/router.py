from fastapi import APIRouter
from .lstm_stress_classifier.prefix_router import router as lstm_stress_classifier_router
from .bertopic.prefix_router import router as bertopic_router


router = APIRouter(
    prefix="/aimodels"
)

router.include_router(lstm_stress_classifier_router)
router.include_router(bertopic_router)
