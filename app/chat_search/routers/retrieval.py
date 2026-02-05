import os
from fastapi import Depends, APIRouter
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session
from mypy_boto3_s3.client import S3Client

from app.aimodels.gpt4all.routers.completions import (
    validate_inputs_and_generate_service,
)
from app.dependencies import get_db, get_s3
from app.aimodels.gpt4all.ai_services.completion_inference import (
    CompletionInference,
    CompletionInferenceInputs,
)
from ..ai_services.service import RetrievalService

router = APIRouter(prefix="", tags=["Query Retrieval - Experimental"])


@router.get(
    "/retrieval",
    response_class=HTMLResponse,
    summary="Query retrieval endpoint",
    response_description="Answerwed query with sources",
)
async def chat_query_retrieval_get(
    prompt: str,
    summarize: bool = False,
    max_docs: int = 1000,
    db: Session = Depends(get_db),
    s3: S3Client = Depends(get_s3),
) -> (HTMLResponse):
    """
    Query retrieval endpoint.
    """

    api_inputs = CompletionInferenceInputs(prompt=prompt)
    completion_inference_service: CompletionInference = (
        validate_inputs_and_generate_service(api_inputs, db, s3)
    )
    retrieval_service = RetrievalService(
        completion_inference=completion_inference_service, db=db, s3=s3
    )

    results = retrieval_service.retrieve(api_inputs, summarize, max_docs=max_docs)

    return _render_result_as_html(results)


# Function to render the result dictionary as HTML
def _render_result_as_html(result):
    # Create a Jinja2 environment and load the template

    env = Environment(
        loader=FileSystemLoader(os.path.abspath(os.path.dirname(__file__))),
        autoescape=True,
    )
    template = env.get_template("template.html")

    # Render the template with the result dictionary
    html_content = template.render(result=result)
    return HTMLResponse(content=html_content)
