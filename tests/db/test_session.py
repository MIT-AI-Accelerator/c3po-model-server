from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker, Session
from app.db.session import engine, SessionLocal
from app.core.config import settings


def test_engine_exists():
    """Test that the database engine is properly created"""
    assert engine is not None
    assert isinstance(engine, Engine)


def test_engine_uses_correct_database_uri():
    """Test that the engine uses the configured database URI"""
    assert engine.url.database == settings.sqlalchemy_database_uri.path.lstrip('/')
    assert engine.url.username == settings.postgres_user
    
    db_uri_str = settings.sqlalchemy_database_uri.unicode_string()
    assert settings.postgres_server in db_uri_str
    assert settings.postgres_db in db_uri_str
    assert str(settings.postgres_port) in db_uri_str


def test_engine_has_pool_pre_ping():
    """Test that the engine is configured with pool_pre_ping"""
    assert engine.pool._pre_ping is True


def test_session_local_is_sessionmaker():
    """Test that SessionLocal is a sessionmaker instance"""
    assert SessionLocal is not None
    assert isinstance(SessionLocal, sessionmaker)


def test_session_local_configuration():
    """Test that SessionLocal is configured correctly"""
    assert SessionLocal.kw.get('autocommit') is False
    assert SessionLocal.kw.get('autoflush') is False


def test_session_local_bound_to_engine():
    """Test that SessionLocal is bound to the correct engine"""
    assert SessionLocal.kw.get('bind') is engine


def test_session_local_creates_session():
    """Test that SessionLocal can create a database session"""
    session = SessionLocal()
    assert session is not None
    assert isinstance(session, Session)
    session.close()


def test_session_can_be_closed():
    """Test that sessions can be properly closed"""
    session = SessionLocal()
    session.close()
    assert True
