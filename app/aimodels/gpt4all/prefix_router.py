from fastapi import APIRouter
from .routers.completions import router as gpt4all_completions_router
from .routers.pretrained import router as gpt4all_pretrained_router

router = APIRouter(
    prefix="/gpt4all", tags=["GPT4All"]
)

router.include_router(gpt4all_completions_router)
router.include_router(gpt4all_pretrained_router)
