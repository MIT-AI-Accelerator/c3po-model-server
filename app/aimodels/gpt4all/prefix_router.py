from fastapi import APIRouter
from .routers.completions import router as bertopic_embedding_pretrained_router
from .routers.documents import router as documents_router
from .routers.completions import router as train_router

router = APIRouter(
    prefix="/gpt4all", tags=["GPT4All"]
)

router.include_router(gpt4all_embedding_pretrained_router)
router.include_router(documents_router)
router.include_router(train_router)
