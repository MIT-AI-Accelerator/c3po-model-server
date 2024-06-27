from app.db.base import Base  # noqa: F401
from .session import engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.schema import DropConstraint, MetaData, Table, ForeignKeyConstraint

# make sure all SQL Alchemy models are imported (app.db.base) before initializing DB
# otherwise, SQL Alchemy might fail to initialize relationships properly
# for more details: https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/28

# add db: Session in here when ready for alembic
def init_db() -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next line

    # pylint: disable=no-member
    Base.metadata.create_all(bind=engine) # noqa: F401

# used to clear the DB (for local / staging, DO NOT USE IN PROD)
def wipe_db() -> None:
    Base.metadata.drop_all(bind=engine) # noqa: F401

def drop_constraints():
    """(On a live db) drops all foreign key constraints before dropping all tables.
    Workaround for SQLAlchemy not doing DROP ## CASCADE for drop_all()
    (https://github.com/pallets/flask-sqlalchemy/issues/722)
    """
    con = engine.connect()
    trans = con.begin()
    inspector = Inspector.from_engine(engine)

    # We need to re-create a minimal metadata with only the required things to
    # successfully emit drop constraints and tables commands for postgres (based
    # on the actual schema of the running instance)
    meta = MetaData()
    tables = []
    all_fkeys = []

    for table_name in inspector.get_table_names():
        fkeys = []

        for fkey in inspector.get_foreign_keys(table_name):
            if not fkey["name"]:
                continue

            fkeys.append(ForeignKeyConstraint((), (), name=fkey["name"]))

        tables.append(Table(table_name, meta, *fkeys))
        all_fkeys.extend(fkeys)

    for fkey in all_fkeys:
        try:
            con.execute(DropConstraint(fkey))
        except InvalidRequestError as err:
            print('unable to drop constraint %s: %s' % (fkey, repr(err)))

    trans.commit()
