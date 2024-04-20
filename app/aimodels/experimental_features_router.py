from fastapi import APIRouter
from .lstm_stress_classifier.prefix_router import router as lstm_stress_classifier_router
from .bertopic.experimental_features_router import router as bertopic_router
from .gpt4all.prefix_router import llm_router, gpt4all_router


router = APIRouter(
    prefix="/aimodels"
)

router.include_router(bertopic_router)
router.include_router(llm_router)
router.include_router(gpt4all_router)
router.include_router(lstm_stress_classifier_router)
