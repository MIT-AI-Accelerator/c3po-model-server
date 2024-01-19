from fastapi import APIRouter
from .aimodels.experimental_features_router import router as aimodels_router
from .sentiments.router import router as sentiments_router
from .chat_search.routers.retrieval import router as chat_queries_router

router = APIRouter()
router.include_router(aimodels_router)
router.include_router(sentiments_router)
router.include_router(chat_queries_router)
