from fastapi import APIRouter
from .routers.upload import router as upload_router

router = APIRouter(
    prefix="/lstmstressclassifier", tags=["LSTM Stress Classifier - Experimental"]
)

router.include_router(upload_router)
