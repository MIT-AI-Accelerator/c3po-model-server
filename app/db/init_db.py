from sqlalchemy.orm import Session
from app.db.base import Base  # noqa: F401
from .session import engine

# make sure all SQL Alchemy models are imported (app.db.base) before initializing DB
# otherwise, SQL Alchemy might fail to initialize relationships properly
# for more details: https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/28

def init_db(db: Session) -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next line

    # pylint: disable=no-member
    Base.metadata.create_all(bind=engine) # noqa: F401
