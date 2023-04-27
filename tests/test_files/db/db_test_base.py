# Import all the models, so that Base has them before creating tables
# pylint: disable=unused-import
from app.db.base import Base  # noqa
from ..crud.crud_test_model import TestModel # noqa
