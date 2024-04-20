from fastapi import APIRouter
from .routers.pretrained import router as llm_pretrained_router
from .routers.completions import router as gpt4all_completions_router

llm_router = APIRouter(
    prefix="/llm", tags=["LLM - Experimental"]
)

llm_router.include_router(llm_pretrained_router)

gpt4all_router = APIRouter(
    prefix="/gpt4all", tags=["GPT4All - Experimental"]
)

gpt4all_router.include_router(gpt4all_completions_router)
