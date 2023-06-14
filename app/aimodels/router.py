from fastapi import APIRouter
from .lstm_stress_classifier.prefix_router import router as lstm_stress_classifier_router
from .bertopic.prefix_router import router as bertopic_router
from .gpt4all.prefix_router import router as gpt4all_router


router = APIRouter(
    prefix="/aimodels"
)

router.include_router(lstm_stress_classifier_router)
router.include_router(bertopic_router)
router.include_router(gpt4all_router)
