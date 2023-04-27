from app.core.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .db_test_base import Base  # noqa: F401

def init_db() -> sessionmaker:
    # connect to PostgreSQL
    engine = create_engine(settings.sqlalchemy_database_uri, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    # pylint: disable=no-member
    Base.metadata.create_all(bind=engine) # noqa: F401

    return SessionLocal

SessionLocal = init_db()
