from fastapi import APIRouter
from .routers.team_trending import router as team_trending_router

router = APIRouter(
    prefix="/insights", tags=["Topics Insights"]
)

router.include_router(team_trending_router)
