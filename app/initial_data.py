import os
import sys
import hashlib
import io
import pickle
import logging
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Union
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from minio.error import InvalidResponseError
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel, EmbeddingModelTypeEnum
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.bertopic.schemas.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate
from app.aimodels.bertopic.schemas.document import DocumentCreate
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllPretrainedModel
from app.aimodels.gpt4all.schemas.gpt4all_pretrained import Gpt4AllPretrainedCreate, Gpt4AllPretrainedUpdate

from app.db.init_db import init_db, wipe_db
from app.db.session import SessionLocal
from app.core.minio import build_client, download_file_from_minio, upload_file_to_minio

from app.core.config import settings, environment_settings
from app.aimodels.bertopic import crud as bertopic_crud
from app.aimodels.gpt4all import crud as gpt4all_crud

from app.mattermost.crud import crud_mattermost
from ppg.mattermost_utils import MM_BOT_USERNAME

from sentence_transformers import SentenceTransformer, CrossEncoder

from sqlalchemy.orm import Session
from minio import Minio
import pandas as pd
from sample_data import CHAT_DATASET_1_PATH
from app.core.model_cache import MODEL_CACHE_BASEDIR

from app.aimodels.bertopic.ai_services.weak_learning import WeakLearner
from sample_data import CHAT_DATASET_4_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    init_db()


def get_db(environment: str, migration_toggle: bool) -> Union[Session, None]:
    db = None

    # clear DB if local or staging as long as not actively testing migrating
    # note: reenabled wipe_db for staging (['local', 'staging']) due to db schema changes, remove staging when schema stable
    if (environment in ['local', 'staging'] and migration_toggle is False):
        logger.info("Clearing database")
        wipe_db()
        logger.info("Database cleared")

    # all environments need to initialize the database
    # prod only if migration toggle is on
    if (environment in ['local', 'development', 'test', 'staging'] or (environment == 'production' and migration_toggle is True)):
        logger.info("Creating database schema and tables")
        db = SessionLocal()
        init()
        logger.info("Initial database schema and tables created.")
    else:
        logger.info("Skipping database initialization")

    return db


def init_minio_bucket(s3: Minio) -> None:
    bucket_name = settings.minio_bucket_name
    try:
        if not s3.bucket_exists(bucket_name):
            s3.make_bucket(bucket_name)
    except InvalidResponseError as err:
        logger.error(err)


def get_s3(environment: str, db: Session) -> Union[Minio, None]:
    s3 = None

    # setup minio client if available (i.e., not in unit tests)
    if (environment in ['local', 'development', 'staging', 'production']):
        logger.info("Connecting MinIO client")
        s3 = build_client()
        logger.info("MinIO client connected")

    if (environment in ['local', 'development']):
        logger.info("Setting up MinIO bucket")
        init_minio_bucket(s3)
        logger.info("MinIO bucket set up.")

    if (environment != 'production'):
        logger.info("Creating documents from chats")
        swagger_string = init_documents_from_chats(db)
        logger.info("Documents created.")
        logger.info(f"Documents: {swagger_string}")

    return s3


def init_sentence_embedding_object(s3: Minio, db: Session, model_path: str) -> None:
    # Create the SentenceTransformer object
    model_name = model_path.split('/')[-1]

    # if model type given, it is here
    model_type = model_path.split('/')[0]

    if model_type == 'cross-encoder':
        embedding_pretrained_model_obj = CrossEncoder(
            model_path, max_length=512)
    else:
        embedding_pretrained_model_obj = SentenceTransformer(model_name)

    # Serialize the object
    serialized_obj = pickle.dumps(embedding_pretrained_model_obj)

    # Calculate the SHA256 hash of the serialized object
    hash_object = hashlib.sha256(serialized_obj)
    hex_dig = hash_object.hexdigest()

    # check to make sure sha256 doesn't already exist
    obj_by_sha: BertopicEmbeddingPretrainedModel = bertopic_crud.bertopic_embedding_pretrained.get_by_sha256(
        db, sha256=hex_dig)

    if not obj_by_sha:
        # Create an in-memory file object
        file_obj = io.BytesIO()

        # Dump the object to the file object
        pickle.dump(embedding_pretrained_model_obj, file_obj)

        # Move the file cursor to the beginning of the file
        file_obj.seek(0)

        if model_type == 'cross-encoder':
            bertopic_embedding_pretrained_obj = BertopicEmbeddingPretrainedCreate(
                sha256=hex_dig, model_name=model_name, model_type=EmbeddingModelTypeEnum.CROSS_ENCODERS)
        else:
            bertopic_embedding_pretrained_obj = BertopicEmbeddingPretrainedCreate(
                sha256=hex_dig, model_name=model_name)

        new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = bertopic_crud.bertopic_embedding_pretrained.create(
            db, obj_in=bertopic_embedding_pretrained_obj)

        # utilize id from above to upload file to minio
        upload_file_to_minio(UploadFile(file_obj),
                             new_bertopic_embedding_pretrained_obj.id, s3)

        # update the object to reflect uploaded status
        updated_object = BertopicEmbeddingPretrainedUpdate(uploaded=True)
        new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = bertopic_crud.bertopic_embedding_pretrained.update(
            db, db_obj=new_bertopic_embedding_pretrained_obj, obj_in=updated_object)

        return new_bertopic_embedding_pretrained_obj

    return obj_by_sha


def init_gpt4all_pretrained_model(s3: Minio, db: Session) -> None:

    model_name = "ggml-gpt4all-l13b-snoozy.bin"
    local_path = os.path.join(
        MODEL_CACHE_BASEDIR, model_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Example model. Check https://github.com/nomic-ai/gpt4all for the latest models.
    url = 'https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin'

    # send a GET request to the URL to download the file. Stream since it's large
    response = requests.get(url, stream=True)

    # open the file in binary mode and write the contents of the response to it in chunks
    # This is a large file, so be prepared to wait.
    hash_object = hashlib.sha256()
    with open(local_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)
                hash_object.update(chunk)

    hex_dig = hash_object.hexdigest()

    # check to make sure sha256 doesn't already exist
    obj_by_sha: Gpt4AllPretrainedModel = gpt4all_crud.gpt4all_pretrained.get_by_sha256(
        db, sha256=hex_dig)

    if not obj_by_sha:

        gpt4all_pretrained_obj = Gpt4AllPretrainedCreate(
            sha256=hex_dig)

        new_gpt4all_pretrained_obj: Gpt4AllPretrainedModel = gpt4all_crud.gpt4all_pretrained.create(
            db, obj_in=gpt4all_pretrained_obj)

        # utilize id from above to upload file to minio
        with open(local_path, 'rb') as file_obj:
            upload_file_to_minio(UploadFile(file_obj),
                                 new_gpt4all_pretrained_obj.id, s3)

        # update the object to reflect uploaded status
        updated_object = Gpt4AllPretrainedUpdate(uploaded=True)
        new_gpt4all_pretrained_obj: Gpt4AllPretrainedModel = gpt4all_crud.gpt4all_pretrained.update(
            db, db_obj=new_gpt4all_pretrained_obj, obj_in=updated_object)

        os.remove(local_path)

        return new_gpt4all_pretrained_obj

    return obj_by_sha


def init_gpt4all_db_obj_staging_prod(s3: Minio, db: Session) -> None:

    model_name = "ggml-gpt4all-l13b-snoozy.bin"
    local_path = os.path.join(
        MODEL_CACHE_BASEDIR, model_name)

    if not os.path.isfile(local_path):
        # Create the directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Download the file from Minio
        logger.info(f"Downloading base model from Minio to {local_path}")
        download_file_from_minio(model_name, s3, filename=local_path)
        logger.info(f"Downloaded model from Minio to {local_path}")

    # check to make sure sha256 doesn't already exist
    obj_by_sha: Gpt4AllPretrainedModel = gpt4all_crud.gpt4all_pretrained.get_by_sha256(
        db, sha256=settings.default_sha256_l13b_snoozy)

    if not obj_by_sha:

        gpt4all_pretrained_obj = Gpt4AllPretrainedCreate(
            sha256=settings.default_sha256_l13b_snoozy, use_base_model=True)

        new_gpt4all_pretrained_obj: Gpt4AllPretrainedModel = gpt4all_crud.gpt4all_pretrained.create(
            db, obj_in=gpt4all_pretrained_obj)

        # update the object to reflect uploaded status
        updated_object = Gpt4AllPretrainedUpdate(uploaded=True)
        new_gpt4all_pretrained_obj: Gpt4AllPretrainedModel = gpt4all_crud.gpt4all_pretrained.update(
            db, db_obj=new_gpt4all_pretrained_obj, obj_in=updated_object)

        return new_gpt4all_pretrained_obj

    return obj_by_sha


def init_weak_learning_object(s3: Minio, db: Session) -> None:
    # Create the weak learner object
    model_name = CHAT_DATASET_4_PATH.split('/')[-1]
    weak_learner_model_obj = WeakLearner().train_weak_learners(CHAT_DATASET_4_PATH)

    # Serialize the object
    serialized_obj = pickle.dumps(weak_learner_model_obj)

    # Calculate the SHA256 hash of the serialized object
    hash_object = hashlib.sha256(serialized_obj)
    hex_dig = hash_object.hexdigest()

    # check to make sure sha256 doesn't already exist
    obj_by_sha: BertopicEmbeddingPretrainedModel = bertopic_crud.bertopic_embedding_pretrained.get_by_sha256(
        db, sha256=hex_dig)

    if not obj_by_sha:
        # Create an in-memory file object
        file_obj = io.BytesIO()

        # Dump the object to the file object
        pickle.dump(weak_learner_model_obj, file_obj)

        # Move the file cursor to the beginning of the file
        file_obj.seek(0)

        bertopic_embedding_pretrained_obj = BertopicEmbeddingPretrainedCreate(
            sha256=hex_dig, model_name=model_name, model_type=EmbeddingModelTypeEnum.WEAK_LEARNERS)

        new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = bertopic_crud.bertopic_embedding_pretrained.create(
            db, obj_in=bertopic_embedding_pretrained_obj)

        # utilize id from above to upload file to minio
        upload_file_to_minio(UploadFile(file_obj),
                             new_bertopic_embedding_pretrained_obj.id, s3)

        # update the object to reflect uploaded status
        updated_object = BertopicEmbeddingPretrainedUpdate(uploaded=True)
        new_bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = bertopic_crud.bertopic_embedding_pretrained.update(
            db, db_obj=new_bertopic_embedding_pretrained_obj, obj_in=updated_object)

        return new_bertopic_embedding_pretrained_obj

    return obj_by_sha


def init_documents_from_chats(db: Session) -> str:
    cfile = CHAT_DATASET_1_PATH
    cdata = pd.read_csv(cfile)
    cdata['messages'] = cdata['messages'].astype(str)
    msgs = cdata['messages'].values.tolist()[6:30]
    documents = [DocumentCreate(text=msg.strip()) for msg in msgs]
    documents_db = bertopic_crud.document.create_all_using_id(
        db, obj_in_list=documents)

    # format for swagger input
    swagger_string = "["
    for document in documents_db:
        swagger_string += f"\"{str(document.id)}\", "

    swagger_string = swagger_string[:-2] + "]"

    return swagger_string


def init_mattermost_bot_user(db: Session, user_name: str) -> None:
    return crud_mattermost.populate_mm_user_team_info(db, user_name=user_name)


def init_large_objects(environment: str, migration_toggle: bool, s3: Minio, db: Session) -> None:

    ########## large object uploads ################
    if (environment == 'local'):
        # Sentence Transformer
        logger.info("Uploading all-MiniLM-L6-v2 object to MinIO")
        embedding_pretrained_obj1 = init_sentence_embedding_object(
            s3, db, "all-MiniLM-L6-v2")
        logger.info("all-MiniLM-L6-v2 object uploaded to MinIO.")
        logger.info(
            f"all-MiniLM-L6-v2 Embedding Pretrained Object ID: {embedding_pretrained_obj1.id}, SHA256: {embedding_pretrained_obj1.sha256}")
        embedding_pretrained_obj2 = init_sentence_embedding_object(
            s3, db, "sentence-transformers/all-mpnet-base-v2")
        logger.info("all-mpnet-base-v2 object uploaded to MinIO.")
        logger.info(
            f"all-mpnet-base-v2 Embedding Pretrained Object ID: {embedding_pretrained_obj2.id}, SHA256: {embedding_pretrained_obj2.sha256}")

        # Weak learner
        logger.info("Uploading WeakLearner object to MinIO")
        embedding_pretrained_obj = init_weak_learning_object(s3, db)
        logger.info("WeakLearner object uploaded to MinIO.")
        logger.info(
            f"Weak learner Pretrained Object ID: {embedding_pretrained_obj.id}, SHA256: {embedding_pretrained_obj.sha256}")

        # MARCO reranker
        logger.info("Uploading MARCO cross-encoder object to MinIO")
        marco_pretrained_obj = init_sentence_embedding_object(
            s3, db, "cross-encoder/ms-marco-TinyBERT-L-6")
        logger.info("MARCO cross-encoder object uploaded to MinIO.")
        logger.info(
            f"MARCO cross-encoder Pretrained Object ID: {marco_pretrained_obj.id}, SHA256: {marco_pretrained_obj.sha256}")

        # Mattermost user
        logger.info("Uploading Mattermost bot user")
        user_obj = init_mattermost_bot_user(db, MM_BOT_USERNAME)
        if user_obj:
            logger.info("Mattermost bot user uploaded to DB")
            logger.info(
                f"Mattermost bot user object ID: {user_obj.id}, mattermost ID: {user_obj.user_id}")
        else:
            logger.info("Unable to load Mattermost bot user")

        # Gpt4All
        logger.info("Uploading Gpt4All object to MinIO")
        gpt4all_pretrained_obj = init_gpt4all_pretrained_model(s3, db)
        logger.info("Gpt4All object uploaded to MinIO.")
        logger.info(
            f"Gpt4All Object ID: {gpt4all_pretrained_obj.id}, SHA256: {gpt4all_pretrained_obj.sha256}")

    if (environment == 'staging' or (environment == 'production' and migration_toggle is True)):
        logger.info("Verifying Gpt4All object in MinIO")
        gpt4all_pretrained_obj = init_gpt4all_db_obj_staging_prod(s3, db)
        logger.info("Verified Gpt4All object in MinIO")
        logger.info(
            f"Gpt4All Object ID: {gpt4all_pretrained_obj.id}, SHA256: {gpt4all_pretrained_obj.sha256}")
    ########## large object uploads ################


def main() -> None:
    args = sys.argv[1:]

    migration_toggle = False
    if len(args) == 1 and args[0] == '--toggle-migration':
        migration_toggle = True

    # environment can be one of 'local', 'test', 'staging', 'production'
    environment = environment_settings.environment

    logger.info(f"Using initialization environment: {environment}")
    logger.info(f"Using migration toggle: {migration_toggle}")

    db = get_db(environment, migration_toggle)

    s3 = get_s3(environment, db)

    init_large_objects(environment, migration_toggle, s3, db)


if __name__ == "__main__":
    main()
