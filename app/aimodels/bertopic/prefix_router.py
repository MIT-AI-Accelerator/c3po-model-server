from fastapi import APIRouter
from .routers.bertopic_embedding_pretrained import router as bertopic_embedding_pretrained_router
from .routers.documents import router as documents_router
from .routers.train import router as train_router

router = APIRouter(
    prefix="/bertopic", tags=["BERTopic"]
)

router.include_router(bertopic_embedding_pretrained_router)
router.include_router(documents_router)
router.include_router(train_router)
