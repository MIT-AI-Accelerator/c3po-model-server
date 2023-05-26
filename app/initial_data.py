import sys
import hashlib
import io
import pickle
import logging
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from minio.error import InvalidResponseError
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.schemas.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate
from app.aimodels.bertopic.schemas.document import DocumentCreate

from app.db.init_db import init_db
from app.db.session import SessionLocal
from app.core.minio import build_client, upload_file_to_minio

from app.core.config import settings, environment_settings
from app.aimodels.bertopic import crud
from sentence_transformers import SentenceTransformer

from sqlalchemy.orm import Session
from minio import Minio
import pandas as pd
from sample_data import CHAT_DATASET_1_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    init_db()


def init_minio_bucket(s3: Minio) -> None:
    bucket_name = settings.minio_bucket_name
    try:
        if not s3.bucket_exists(bucket_name):
            s3.make_bucket(bucket_name)
    except InvalidResponseError as err:
        print(err)


def init_sentence_embedding_object(s3: Minio, db: Session) -> None:
    # Create the SentenceTransformer object
    model_name = "all-MiniLM-L6-v2"
    embedding_pretrained_model_obj = SentenceTransformer(model_name)

    # Serialize the object
    serialized_obj = pickle.dumps(embedding_pretrained_model_obj)

    # Calculate the SHA256 hash of the serialized object
    hash_object = hashlib.sha256(serialized_obj)
    hex_dig = hash_object.hexdigest()

    # check to make sure sha256 doesn't already exist
    obj_by_sha: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get_by_sha256(
        db, sha256=hex_dig)

    if not obj_by_sha:
        # Create an in-memory file object
        file_obj = io.BytesIO()

        # Dump the object to the file object
        pickle.dump(embedding_pretrained_model_obj, file_obj)

        # Move the file cursor to the beginning of the file
        file_obj.seek(0)

        bertopic_embedding_pretrained_obj = BertopicEmbeddingPretrainedCreate(
            sha256=hex_dig, model_name=model_name)

        new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.create(
            db, obj_in=bertopic_embedding_pretrained_obj)

        # utilize id from above to upload file to minio
        upload_file_to_minio(UploadFile(file_obj),
                             new_bertopic_embedding_pretrained_obj.id, s3)

        # update the object to reflect uploaded status
        updated_object = BertopicEmbeddingPretrainedUpdate(uploaded=True)
        new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.update(
            db, db_obj=new_bertopic_embedding_pretrained_obj, obj_in=updated_object)

        return new_bertopic_embedding_pretrained_obj

    return obj_by_sha


def init_documents_from_chats(db: Session) -> str:
    cfile = CHAT_DATASET_1_PATH
    cdata = pd.read_csv(cfile)
    cdata['messages'] = cdata['messages'].astype(str)
    msgs = cdata['messages'].values.tolist()[6:30]
    documents = [DocumentCreate(text=msg.strip()) for msg in msgs]
    documents_db = crud.document.create_all_using_id(db, obj_in_list=documents)

    # format for swagger input
    swagger_string = "["
    for document in documents_db:
        swagger_string += f"\"{str(document.id)}\", "

    swagger_string = swagger_string[:-2] + "]"

    return swagger_string


def main() -> None:
    args = sys.argv[1:]

    migration_toggle = False
    if len(args) == 1 and args[0] == '--toggle-migration':
        migration_toggle = True

    # environment can be one of 'local', 'test', 'staging', 'production'
    environment = environment_settings.environment

    logger.info(f"Using initialization environment: {environment}")
    logger.info(f"Using migration toggle: {migration_toggle}")

    # all environments need to initialize the database
    # prod only if migration toggle is on
    if (environment in ['local', 'test', 'staging'] or (environment == 'production' and migration_toggle is True)):
        logger.info("Creating database schema and tables")
        db = SessionLocal()
        init()
        logger.info("Initial database schema and tables created.")
    else:
        logger.info("Skipping database initialization")

    if (environment == 'local'):
        logger.info("Setting up MinIO bucket")
        s3 = build_client()
        init_minio_bucket(s3)
        logger.info("MinIO bucket set up.")

    if (environment == 'local'):
        logger.info("Uploading SentenceTransformer object to MinIO")
        embedding_pretrained_obj = init_sentence_embedding_object(s3, db)
        logger.info("SentenceTransformer object uploaded to MinIO.")
        logger.info(
            f"Embedding Pretrained Object ID: {embedding_pretrained_obj.id}, SHA256: {embedding_pretrained_obj.sha256}")

    if (environment != 'production'):
        logger.info("Creating documents from chats")
        swagger_string = init_documents_from_chats(db)
        logger.info("Documents created.")
        logger.info(f"Documents: {swagger_string}")


if __name__ == "__main__":
    main()
