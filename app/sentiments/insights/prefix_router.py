from fastapi import APIRouter
from .routers.get_chat_stress import router as get_chat_stress_router

router = APIRouter(
    prefix="/insights", tags=["Sentiment Insights - Experimental"]
)

router.include_router(get_chat_stress_router)
