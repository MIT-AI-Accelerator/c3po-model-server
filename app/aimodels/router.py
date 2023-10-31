from fastapi import APIRouter
from .bertopic.prefix_router import router as bertopic_router


router = APIRouter(
    prefix="/aimodels"
)

router.include_router(bertopic_router)
