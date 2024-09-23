import os
import sys
import hashlib
import io
import pickle
import logging
import requests
import time
from pathlib import Path
from tqdm import tqdm
from typing import Union
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from minio.error import InvalidResponseError

import ppg.services.mattermost_utils as mattermost_utils
from ppg.schemas.bertopic.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedCreate, BertopicEmbeddingPretrainedUpdate
from ppg.schemas.gpt4all.llm_pretrained import LlmPretrainedCreate, LlmPretrainedUpdate
from ppg.schemas.bertopic.document import DocumentCreate
from ppg.schemas.mattermost.mattermost_documents import ThreadTypeEnum

from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel, EmbeddingModelTypeEnum
from app.aimodels.bertopic.models.document import DocumentModel
from app.aimodels.gpt4all.models.llm_pretrained import LlmPretrainedModel, LlmFilenameEnum

from app.db.init_db import init_db, wipe_db, drop_constraints
from app.db.session import SessionLocal
from app.core.minio import build_client, download_file_from_minio, upload_file_to_minio, list_minio_objects

from app.core.config import settings, environment_settings, get_label_dictionary
from app.aimodels.bertopic import crud as bertopic_crud
from app.aimodels.gpt4all import crud as gpt4all_crud

from app.mattermost.crud import crud_mattermost
from app.mattermost.models.mattermost_users import MattermostUserModel

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
        drop_constraints()
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


def init_sentence_embedding_object(s3: Minio, db: Session, model_path: str) -> BertopicEmbeddingPretrainedModel:
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
                sha256=hex_dig, model_name=model_name, model_type=EmbeddingModelTypeEnum.SENTENCE_TRANSFORMERS)

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


def init_mistrallite_pretrained_model(s3: Minio, db: Session) -> LlmPretrainedModel:

    model_name = "mistrallite.Q4_K_M.gguf"
    local_path = os.path.join(
        MODEL_CACHE_BASEDIR, model_name)

    if os.path.isfile(local_path):
        logger.info(f"Local {model_name} found")

        with open(local_path,"rb") as f:
            bin_data = f.read()
            hash_object = hashlib.sha256(bin_data)

    else:
        logger.info(f"Downloading {model_name}")

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Example model. Check https://github.com/nomic-ai/gpt4all for the latest models.
        url = 'https://huggingface.co/TheBloke/MistralLite-7B-GGUF/resolve/main/mistrallite.Q4_K_M.gguf?download=true'

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
    obj_by_sha: LlmPretrainedModel = gpt4all_crud.llm_pretrained.get_by_sha256(
        db, sha256=hex_dig)

    if not obj_by_sha:

        llm_pretrained_obj = LlmPretrainedCreate(
            sha256=hex_dig, model_type=LlmFilenameEnum.Q4_K_M)

        new_llm_pretrained_obj: LlmPretrainedModel = gpt4all_crud.llm_pretrained.create(
            db, obj_in=llm_pretrained_obj)

        # utilize id from above to upload file to minio
        with open(local_path, 'rb') as file_obj:
            upload_file_to_minio(UploadFile(file_obj),
                                 new_llm_pretrained_obj.id, s3)

        # update the object to reflect uploaded status
        updated_object = LlmPretrainedUpdate(uploaded=True)
        new_llm_pretrained_obj: LlmPretrainedModel = gpt4all_crud.llm_pretrained.update(
            db, db_obj=new_llm_pretrained_obj, obj_in=updated_object)

        return new_llm_pretrained_obj

    return obj_by_sha


def init_llm_pretrained_model(s3: Minio, db: Session) -> LlmPretrainedModel:

    model_name = "ggml-gpt4all-l13b-snoozy.bin"
    local_path = os.path.join(
        MODEL_CACHE_BASEDIR, model_name)

    if os.path.isfile(local_path):
        logger.info(f"Local {model_name} found")

        with open(local_path,"rb") as f:
                bin_data = f.read()
                hash_object = hashlib.sha256(bin_data)

    else:
        logger.info(f"Downloading {model_name}")

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
    obj_by_sha: LlmPretrainedModel = gpt4all_crud.llm_pretrained.get_by_sha256(
        db, sha256=hex_dig)

    if not obj_by_sha:

        llm_pretrained_obj = LlmPretrainedCreate(
            sha256=hex_dig)

        new_llm_pretrained_obj: LlmPretrainedModel = gpt4all_crud.llm_pretrained.create(
            db, obj_in=llm_pretrained_obj)

        # utilize id from above to upload file to minio
        with open(local_path, 'rb') as file_obj:
            upload_file_to_minio(UploadFile(file_obj),
                                 new_llm_pretrained_obj.id, s3)

        # update the object to reflect uploaded status
        updated_object = LlmPretrainedUpdate(uploaded=True)
        new_llm_pretrained_obj: LlmPretrainedModel = gpt4all_crud.llm_pretrained.update(
            db, db_obj=new_llm_pretrained_obj, obj_in=updated_object)

        return new_llm_pretrained_obj

    return obj_by_sha


def init_llm_db_obj_staging_prod(s3: Minio, db: Session, model_enum: LlmFilenameEnum) -> LlmPretrainedModel:

    default_sha256 = settings.default_sha256_l13b_snoozy \
        if model_enum == LlmFilenameEnum.L13B_SNOOZY \
        else settings.default_sha256_q4_k_m
    model_name = model_enum.value
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
    obj_by_sha: LlmPretrainedModel = gpt4all_crud.llm_pretrained.get_by_sha256(
        db, sha256=default_sha256)

    if not obj_by_sha:

        llm_pretrained_obj = LlmPretrainedCreate(
            sha256=default_sha256, use_base_model=True)

        new_llm_pretrained_obj: LlmPretrainedModel = gpt4all_crud.llm_pretrained.create(
            db, obj_in=llm_pretrained_obj)

        # update the object to reflect uploaded status
        updated_object = LlmPretrainedUpdate(uploaded=True)
        new_llm_pretrained_obj: LlmPretrainedModel = gpt4all_crud.llm_pretrained.update(
            db, db_obj=new_llm_pretrained_obj, obj_in=updated_object)

        return new_llm_pretrained_obj

    return obj_by_sha


def init_weak_learning_object(s3: Minio, db: Session) -> BertopicEmbeddingPretrainedModel:
    # Create the weak learner object
    model_name = CHAT_DATASET_4_PATH.split('/')[-1]
    df_train = pd.read_csv(CHAT_DATASET_4_PATH)
    weak_learner_model_obj = WeakLearner().train_weak_learners(df_train)

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
            sha256=hex_dig, model_name=model_name, model_type=EmbeddingModelTypeEnum.WEAK_LEARNERS, reference=get_label_dictionary())

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


def init_mattermost_bot_user(db: Session, user_name: str) -> MattermostUserModel:
    return crud_mattermost.populate_mm_user_team_info(db, user_name=user_name, get_teams=True)


def init_mattermost_documents(db:Session, bot_obj: MattermostUserModel) -> None:
        cdf = mattermost_utils.get_all_user_team_channels(settings.mm_base_url,
            settings.mm_token,
            bot_obj.user_id,
            bot_obj.teams)
        channel_ids = cdf.id.values
        logger.info('found %d nitmre-bot channels' % len(channel_ids))

        # nitmre-bot may be a member of 1000s of channels, this may take a lot of time
        # start with 2 channels for now: fm_618aoc, jicc618aoc
        channel_ids = ["49qb17rn4pyxzf8t7tn5q5i9by", "rzmytnht33fjxr7dy46p8aqb9e"]
        history_depth = 0
        filter_system_posts = True

        adf = pd.DataFrame()
        for channel_id in tqdm(channel_ids):
            channel_obj = crud_mattermost.get_or_create_mm_channel_object(db, channel_id=channel_id)
            df = mattermost_utils.get_channel_posts(
                settings.mm_base_url,
                settings.mm_token,
                channel_id,
                history_depth=history_depth,
                filter_system_types=filter_system_posts).assign(channel=channel_obj.id)
            adf = pd.concat([adf, df], ignore_index=True)
        channel_uuids = adf['channel'].unique()

        if not adf.empty:
            user_ids = adf['user_id'].unique()
            for uid in user_ids:
                user_obj = crud_mattermost.get_or_create_mm_user_object(db, user_id=uid)
                adf.loc[adf['user_id'] == uid, 'user'] = user_obj.id

            channel_document_objs = crud_mattermost.mattermost_documents.get_all_channel_documents(
                db, channels=channel_uuids)
            existing_ids = [obj.message_id for obj in channel_document_objs]
            adf = adf[~adf.id.isin(existing_ids)].drop_duplicates(subset='id')

        adf.rename(columns={'id': 'message_id'}, inplace=True)
        return crud_mattermost.mattermost_documents.create_all_using_df(db, ddf=adf, thread_type=ThreadTypeEnum.MESSAGE)

########## large object uploads ################

def init_large_objects(db: Session) -> None:

    # Mattermost user
    logger.info("Uploading Mattermost bot user")
    bot_obj = init_mattermost_bot_user(db, mattermost_utils.MM_BOT_USERNAME)
    if bot_obj:
        logger.info("Mattermost bot user uploaded to DB")
        logger.info(
            f"Mattermost bot user object ID: {bot_obj.id}, mattermost ID: {bot_obj.user_id}")
    else:
        logger.info("Unable to load Mattermost bot user")

    # Mattermost documents
    logger.info("Uploading Mattermost documents")
    doc_objs = init_mattermost_documents(db, bot_obj)
    if doc_objs:
        logger.info("Uploaded %d Mattermost documents to DB" % len(doc_objs))
    else:
        logger.info("Unable to load Mattermost documents")


def init_large_objects_local(s3: Minio, db: Session) -> None:

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

    # Mistral
    logger.info("Uploading Mistral object to MinIO")
    llm_pretrained_obj = init_mistrallite_pretrained_model(s3, db)
    logger.info("Mistral object uploaded to MinIO.")
    logger.info(
        f"Mistral Object ID: {llm_pretrained_obj.id}, SHA256: {llm_pretrained_obj.sha256}")

    # Gpt4All
    logger.info("Uploading Gpt4All object to MinIO")
    llm_pretrained_obj = init_llm_pretrained_model(s3, db)
    logger.info("Gpt4All object uploaded to MinIO.")
    logger.info(
        f"Gpt4All Object ID: {llm_pretrained_obj.id}, SHA256: {llm_pretrained_obj.sha256}")

def init_large_objects_p1(s3: Minio, db: Session) -> None:

    # Mistral
    logger.info("Verifying Mistral object in MinIO")
    llm_pretrained_obj = init_llm_db_obj_staging_prod(s3, db, LlmFilenameEnum.Q4_K_M)
    logger.info("Verified Mistral object in MinIO.")
    logger.info(
        f"Mistral Object ID: {llm_pretrained_obj.id}, SHA256: {llm_pretrained_obj.sha256}")

    # Gpt4All
    logger.info("Verifying Gpt4All object in MinIO")
    llm_pretrained_obj = init_llm_db_obj_staging_prod(s3, db, LlmFilenameEnum.L13B_SNOOZY)
    logger.info("Verified Gpt4All object in MinIO.")
    logger.info(
        f"Gpt4All Object ID: {llm_pretrained_obj.id}, SHA256: {llm_pretrained_obj.sha256}")

########## large object uploads ################


def main() -> None:
    start = time.time()

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

    if (environment != 'test'):
        init_large_objects(db)

    if (environment == 'local'):
        init_large_objects_local(s3, db)
    elif (environment == 'staging' or (environment == 'production' and migration_toggle is True)):
        init_large_objects_p1(s3, db)

    list_minio_objects(s3)

    end = time.time()
    logger.info("Initialization complete in %fs" % (end - start))


if __name__ == "__main__":
    main()
