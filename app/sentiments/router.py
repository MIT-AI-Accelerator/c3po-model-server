from fastapi import APIRouter
from .insights.prefix_router import router as insights_router

router = APIRouter(
    prefix="/sentiments"
)

router.include_router(insights_router)
