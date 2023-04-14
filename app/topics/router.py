from fastapi import APIRouter
from .insights.prefix_router import router as insights_router

router = APIRouter(
    prefix="/topics"
)

router.include_router(insights_router)
