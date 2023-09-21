from uuid import uuid4
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import UUID
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import select, update

from app.db.base_class import Base
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm.exc import UnmappedInstanceError

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: type[ModelType]):
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).

        **Parameters**

        * `model`: A SQLAlchemy model class
        * `schema`: A Pydantic model (schema) class
        """
        self.model = model

    # *************** standard from FastAPI examples*********************
    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        obj_data = jsonable_encoder(db_obj)
        if not obj_data:
            db.refresh(db_obj)
            obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])

        stmt = (update(self.model).
                where(self.model.id == db_obj.id).
                values(**jsonable_encoder(db_obj))
        )

        db.execute(stmt)

        try:
            db.commit() # see here for why we use commit: https://stackoverflow.com/questions/41624316/when-do-commits-happen-with-sqlalchemy-core
            db.refresh(db_obj)
        except InvalidRequestError:  # id is not in db
            db.rollback()
            return None
        return db_obj

    def remove(self, db: Session, *, id: Any) -> ModelType:
        obj = db.query(self.model).get(id)
        try:
            db.delete(obj)
        except UnmappedInstanceError:
            db.rollback()
            return None
        db.commit()
        return obj

    # *************** additional custom methods *********************
    # see here for using populate_existing for multi-refresh
    # https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-populate-existing
    def refresh_all_by_id(self, db: Session, *, db_obj_ids: list[UUID]) -> list[ModelType]:
        if not db_obj_ids:
            return []
        stmt = (
            select(self.model)
            .where(self.model.id.in_(db_obj_ids))
            .execution_options(populate_existing=True)
        )
        return db.execute(stmt).scalars().all() # no mutation so no need to commit

    def create_all_using_id(self, db: Session, *, obj_in_list: list[CreateSchemaType]) -> list[ModelType]:
        if not obj_in_list:
            return []
        db_obj_list = [self.model(**jsonable_encoder(obj_in), id=uuid4())
                       for obj_in in obj_in_list]  # type: ignore
        db_obj_ids = [db_obj.id for db_obj in db_obj_list]
        db.add_all(db_obj_list)
        db.commit()
        return self.refresh_all_by_id(db, db_obj_ids=db_obj_ids)
