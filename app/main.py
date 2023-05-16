

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_versioning import VersionedFastAPI
from .core.config import settings
from .core.logging import logger
from .aimodels.router import router as aimodels_router
from .sentiments.router import router as sentiments_router
from .topics.router import router as topics_router
from .dependencies import httpx_client

logger.info("Dummy Info")
logger.error("Dummy Error")
logger.debug("Dummy Debug")
logger.warning("Dummy Warning")
logger.info("UI Root: %s", settings.docs_ui_root_path)
logger.info("log_level: %s", settings.log_level)
logger.warning("Test filtering this_should_be_filtered_out")

# initiate the app and tell it that there is a proxy prefix of /api that gets stripped
# (only effects the loading of the swagger and redoc UIs)
app = FastAPI(title="Transformers API", root_path=settings.docs_ui_root_path,
              responses={404: {"description": "Not found"}})

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3004",
    "https://transformers.staging.dso.mil",
    "https://transformers.apps.dso.mil",
]

app.include_router(aimodels_router)
app.include_router(sentiments_router)
app.include_router(topics_router)

# setup for major versioning
# ensure to copy over all the non-title args to the original FastAPI call...read docs here: https://pypi.org/project/fastapi-versioning/
versioned_app = VersionedFastAPI(app,
                                 version_format='{major}',
                                 prefix_format='/v{major}', default_api_version=1, root_path=settings.docs_ui_root_path)

# add middleware here since this is the app that deploys
versioned_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# close the httpx client when app is shutdown
# see here: https://stackoverflow.com/questions/73721736/what-is-the-proper-way-to-make-downstream-https-requests-inside-of-uvicorn-fasta
@versioned_app.on_event('shutdown')
async def shutdown_event():
    await httpx_client.aclose()
