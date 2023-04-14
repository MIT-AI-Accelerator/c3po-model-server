
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_versioning import VersionedFastAPI

from .settings.settings import settings
from .aimodels.router import router as aimodels_router
from .sentiments.router import router as sentiments_router
from .topics.router import router as topics_router



# initiate the app and tell it that there is a proxy prefix of /api that gets stripped
# (only effects the loading of the swagger and redoc UIs)
app = FastAPI(title="Transformers API", root_path=settings.docs_ui_root_path,
              responses={404: {"description": "Not found"}})

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3004",
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
