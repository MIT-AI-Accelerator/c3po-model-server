from fastapi import APIRouter
from .routers.summarization import router as summary_router

router = APIRouter(
    prefix="/bertopic", tags=["BERTopic - External"]
)

router.include_router(summary_router)
