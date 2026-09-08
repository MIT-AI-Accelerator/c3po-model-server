from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

if settings.sqlalchemy_database_uri is None:
    raise ValueError("sqlalchemy_database_uri is not configured. Please set SQLALCHEMY_DATABASE_URI environment variable.")

engine = create_engine(settings.sqlalchemy_database_uri.unicode_string(), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
