from typing import Union, Optional
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
from sqlalchemy.orm import Session
from mypy_boto3_s3.client import S3Client
import pandas as pd
from app.core.logging import logger
from app.core.s3 import pickle_and_upload_object_to_s3
from app.core.errors import ValidationError, HTTPValidationError
from app.core.config import settings, get_acronym_dictionary, get_icao_dictionary
from app.dependencies import get_db, get_s3
from app.ppg_common.schemas.bertopic.document_embedding_computation import DocumentEmbeddingComputationCreate
from app.ppg_common.schemas.bertopic.bertopic_trained import BertopicTrained, BertopicTrainedCreate, BertopicTrainedUpdate
from app.ppg_common.schemas.bertopic.bertopic_visualization import BertopicVisualizationCreate
from app.ppg_common.schemas.bertopic.topic import TopicSummaryUpdate
from app.nitmre_nlp_utils.preprocess import preprocess_message
from app.aimodels.gpt4all.models import LlmPretrainedModel
from app.aimodels.gpt4all.crud import crud_llm_pretrained
import app.mattermost.crud.crud_mattermost as crud_mattermost
from .. import crud
from ..ai_services.basic_inference import BasicInference, MIN_BERTOPIC_DOCUMENTS, DEFAULT_TRAIN_PERCENT
from ..models.bertopic_trained import BertopicTrainedModel
from ..models.bertopic_visualization import BertopicVisualizationTypeEnum
from ..ai_services.topic_summarization import DEFAULT_TREND_DEPTH_DAYS, DEFAULT_PROMPT_TEMPLATE, DEFAULT_REFINE_TEMPLATE
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel

router = APIRouter(
    prefix=""
)


class TrainModelRequest(BaseModel):
    sentence_transformer_id: UUID4
    weak_learner_id: Optional[UUID4 | None] = None
    summarization_model_id: Optional[UUID4 | None] = None
    document_ids: list[UUID4] = []
    summarization_document_ids: Optional[list[UUID4]] = []
    num_topics: Optional[int] = 2
    seed_topics: Optional[list[list]] = []
    stop_words: Optional[list[str]] = []
    trends_only: Optional[bool] = False
    trend_depth: Optional[int] = DEFAULT_TREND_DEPTH_DAYS
    train_percent: Optional[float] = DEFAULT_TRAIN_PERCENT
    prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    refine_template: Optional[str] = DEFAULT_REFINE_TEMPLATE


@router.post(
    "/model/train",
    response_model=Union[BertopicTrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Train BERTopic on text",
    response_description="Trained Model and Plotly Visualization config"
)
def train_bertopic_post(request: TrainModelRequest, db: Session = Depends(get_db), s3: S3Client = Depends(get_s3)) -> (
    Union[BertopicTrained, HTTPValidationError]
):
    """
    Train a BERTopic model on text.
    """
    # verify enough documents to actually handle clustering
    if len(request.document_ids) < MIN_BERTOPIC_DOCUMENTS:
        raise HTTPException(
            status_code=400, detail="must have at least 7 documents to find topics")
    elif len(request.summarization_document_ids) == 0 or len(request.document_ids) != len(request.summarization_document_ids):
        logger.warning("reusing document_ids for summarization, length mismatch")
        request.summarization_document_ids = request.document_ids

    # validate train percent
    if request.train_percent < 0.0 or request.train_percent > 1.0:
        raise HTTPException(
            status_code=400, detail="train pecent must be between [0, 1]")

    documents, precalculated_embeddings = get_documents_and_embeddings(db, request.document_ids, request.sentence_transformer_id)
    document_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids=request.document_ids)
    summarization_document_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids=request.summarization_document_ids)

    if all(document_df['message_id'].str.len() > 0):
        # merge original messages with summarization text
        document_df = pd.merge(document_df,
                            summarization_document_df[['message_id', 'message']],
                            on='message_id',
                            how='left',
                            validate='1:1').rename(columns={"message_x": "message",
                                                            "message_y": "summarization_message"})
        fix_mask = document_df['summarization_message'].isnull()
        document_df.summarization_message[fix_mask] = document_df.message[fix_mask]
    else:
        logger.warning("reusing document_ids for summarization, missing merge criteria")
        document_df['summarization_message'] = document_df['message']
    document_df['summarization_message'] = document_df.apply(lambda r: preprocess_message(get_acronym_dictionary(), get_icao_dictionary(), r['summarization_message'], msg_only=True), axis=1)

    # train the model
    basic_inference = validate_inference_inputs_and_generate_service(request, db, s3)
    inference_output = basic_inference.train_bertopic_on_documents(db,
                                                                   documents, precalculated_embeddings=precalculated_embeddings, num_topics=request.num_topics,
                                                                   document_df=document_df,
                                                                   seed_topic_list=request.seed_topics,
                                                                   trends_only=request.trends_only,
                                                                   trend_depth=request.trend_depth,
                                                                   train_percent=request.train_percent)

    # save calculated embeddings computations
    new_embedding_computation_obj_list = [DocumentEmbeddingComputationCreate(
        document_id=documents[i].id,
        bertopic_embedding_pretrained_id=request.sentence_transformer_id,
        embedding_vector=inference_output.embeddings[i],
        originated_from=settings.originated_from
    ) for i, wasUpdated in enumerate(inference_output.updated_document_indicies) if wasUpdated]

    crud.document_embedding_computation.create_all_using_id(
        db, obj_in_list=new_embedding_computation_obj_list)

    # refresh the documents
    documents = crud.document.refresh_all_by_id(
        db, db_obj_ids=request.document_ids)

    # create and save a trained model object
    bertopic_trained_obj = BertopicTrainedCreate(
        sentence_transformer_id=request.sentence_transformer_id,
        weak_learner_id=request.weak_learner_id,
        summarization_model_id=request.summarization_model_id,
        seed_topics=pd.DataFrame({'seed_topics': request.seed_topics})[
            'seed_topics'].to_dict(),
        stop_words=pd.DataFrame({'stop_words': request.stop_words})[
            'stop_words'].to_dict(),
        prompt_template=request.prompt_template,
        refine_template=request.refine_template,
        uploaded=False
    )

    # save the trained model object in the database
    new_bertopic_trained_obj: BertopicTrainedModel = crud.bertopic_trained.create_with_embedding_pretrained_id(
        db, obj_in=bertopic_trained_obj, embedding_pretrained_id=request.sentence_transformer_id)

    # upload the trained model to s3
    upload_success = pickle_and_upload_object_to_s3(
        object=inference_output.topic_model, id=new_bertopic_trained_obj.id, s3=s3)

    # if upload was successful, set the uploaded flag to true in the database using crud.bertopic_trained.update
    if upload_success:
        new_bertopic_trained_obj = crud.bertopic_trained.update(
            db, db_obj=new_bertopic_trained_obj, obj_in=BertopicTrainedUpdate(uploaded=True))

    upload_topics_and_visualizations(db, new_bertopic_trained_obj.id, inference_output)

    # save the join table between the documents and the trained model
    # see here: https://docs.sqlalchemy.org/en/20/orm/basic_relationships.html#many-to-many
    # and here: https://stackoverflow.com/questions/25668092/flask-sqlalchemy-many-to-many-insert-data
    new_bertopic_trained_obj.trained_on_documents.extend(documents)
    db.add(new_bertopic_trained_obj)
    db.commit()

    # refresh the new trained model object model
    # (see docs here for info: https://docs.sqlalchemy.org/en/20/orm/session_state_management.html#refreshing-expiring)
    db.refresh(new_bertopic_trained_obj)

    return new_bertopic_trained_obj


def validate_obj(obj: Union[BertopicEmbeddingPretrainedModel, None]):
    if not obj:
        raise HTTPException(
            status_code=422, detail=f"Invalid {str(obj.model_type)} id")

    if not obj.uploaded:
        raise HTTPException(
            status_code=422, detail=f"{str(obj.model_type)} pretrained model not uploaded")


def validate_inference_inputs_and_generate_service(request: TrainModelRequest, db: Session, s3: S3Client):

    # check to make sure id exists
    bertopic_sentence_transformer_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get(
        db, request.sentence_transformer_id)

    bertopic_weak_learner_obj = None
    if request.weak_learner_id:
        # check to make sure id exists
        bertopic_weak_learner_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get(
            db, request.weak_learner_id)

        validate_obj(bertopic_weak_learner_obj)

    # disabling summarization due to issue with pydantic v2 validation of langchain_community.llms.CTransformers
    # https://github.com/orgs/MIT-AI-Accelerator/projects/3/views/2?pane=issue&itemId=91139209
    request.summarization_model_id = None
    llm_pretrained_obj = None
    if request.summarization_model_id:
        # check to make sure id exists
        llm_pretrained_obj: LlmPretrainedModel = crud_llm_pretrained.llm_pretrained.get(
            db, request.summarization_model_id)

        validate_obj(llm_pretrained_obj)

    validate_obj(bertopic_sentence_transformer_obj)

    return BasicInference(bertopic_sentence_transformer_obj,
                            s3,
                            request.prompt_template,
                            request.refine_template,
                            bertopic_weak_learner_obj,
                            llm_pretrained_obj,
                            stop_word_list=request.stop_words)


def get_documents_and_embeddings(db, document_ids, sentence_transformer_id):

    # get the documents
    documents = []
    for document_id in document_ids:
        document_obj = crud.document.get(db, document_id)
        if not document_obj or document_obj.original_created_time is None:
            raise HTTPException(status_code=422, detail=f'Invalid document id {document_id}')
        documents.append(document_obj)

    # sort by timestamp necessary for train/test data split
    documents.sort(key=lambda x: x.original_created_time)

    # extract any formerly computed embeddings, needs to be list[Union[list[float], None]]
    # use None if no embedding was computed previously
    precalculated_embeddings = []
    for document in documents:

        next_value = None
        for embedding_computation in document.embedding_computations:
            if embedding_computation.bertopic_embedding_pretrained_id == sentence_transformer_id:
                next_value = embedding_computation.embedding_vector
                break

        precalculated_embeddings.append(next_value)

    return documents, precalculated_embeddings

def upload_topics_and_visualizations(db, model_id, inference_output):

    # upload model-level visualizations
    visualize_model_words = BertopicVisualizationCreate(
        model_or_topic_id=model_id,
        visualization_type=BertopicVisualizationTypeEnum.MODEL_WORDS,
        html_string=inference_output.model_word_visualization.to_html(),
        json_string=inference_output.model_word_visualization.to_json()
    )
    crud.bertopic_visualization.create(db, obj_in=visualize_model_words)
    visualize_model_clusters = BertopicVisualizationCreate(
        model_or_topic_id=model_id,
        visualization_type=BertopicVisualizationTypeEnum.MODEL_CLUSTERS,
        html_string=inference_output.model_cluster_visualization.to_html(),
        json_string=inference_output.model_cluster_visualization.to_json()
    )
    crud.bertopic_visualization.create(db, obj_in=visualize_model_clusters)
    visualize_model_timeline = BertopicVisualizationCreate(
        model_or_topic_id=model_id,
        visualization_type=BertopicVisualizationTypeEnum.MODEL_TIMELINE,
        html_string=inference_output.model_timeline_visualization.to_html(),
        json_string=inference_output.model_timeline_visualization.to_json()
    )
    crud.bertopic_visualization.create(db, obj_in=visualize_model_timeline)

    # upload topics and topic-level visualizations
    if len(inference_output.topics) != len(inference_output.topic_timeline_visualization):
            raise HTTPException(
            status_code=422, detail="topic summary and visualization length mismatch")

    for tid in range(len(inference_output.topics)):
        crud.topic_summary.update(db,
                                  db_obj=inference_output.topics[tid],
                                  obj_in=TopicSummaryUpdate(model_id=model_id))

        visualize_topic_timeline = BertopicVisualizationCreate(
            model_or_topic_id=inference_output.topics[tid].id,
            visualization_type=BertopicVisualizationTypeEnum.TOPIC_TIMELINE,
            html_string=inference_output.topic_timeline_visualization[tid].to_html(),
            json_string=inference_output.topic_timeline_visualization[tid].to_json()
        )
        crud.bertopic_visualization.create(db, obj_in=visualize_topic_timeline)
