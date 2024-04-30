from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from minio import Minio
from ppg.schemas.bertopic.document_embedding_computation import DocumentEmbeddingComputationCreate
from ppg.schemas.bertopic.bertopic_trained import BertopicTrained, BertopicTrainedCreate, BertopicTrainedUpdate
from ppg.schemas.bertopic.bertopic_visualization import BertopicVisualizationCreate
from ppg.schemas.bertopic.topic import TopicSummaryUpdate
from app.core.minio import pickle_and_upload_object_to_minio
from ..ai_services.basic_inference import BasicInference, MIN_BERTOPIC_DOCUMENTS
from app.dependencies import get_db, get_minio
from .. import crud
from ..models.bertopic_trained import BertopicTrainedModel
from ..models.bertopic_visualization import BertopicVisualizationTypeEnum
from ..ai_services.topic_summarization import DEFAULT_TREND_DEPTH_DAYS, DEFAULT_PROMPT_TEMPLATE, DEFAULT_REFINE_TEMPLATE
from app.core.errors import ValidationError, HTTPValidationError
from app.core.config import settings
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.aimodels.gpt4all.models import LlmPretrainedModel
from app.aimodels.gpt4all.crud import crud_llm_pretrained
import app.mattermost.crud.crud_mattermost as crud_mattermost

router = APIRouter(
    prefix=""
)


class TrainModelRequest(BaseModel):
    sentence_transformer_id: UUID4
    weak_learner_id: UUID4 | None
    summarization_model_id: UUID4 | None
    document_ids: list[UUID4] = []
    num_topics: int = 2
    seed_topics: list[list] = []
    stop_words: list[str] = []
    trends_only: bool = False
    trend_depth: int = DEFAULT_TREND_DEPTH_DAYS
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    refine_template: str = DEFAULT_REFINE_TEMPLATE


@router.post(
    "/model/train",
    response_model=Union[BertopicTrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Train BERTopic on text",
    response_description="Trained Model and Plotly Visualization config"
)
def train_bertopic_post(request: TrainModelRequest, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    Union[BertopicTrained, HTTPValidationError]
):
    """
    Train a BERTopic model on text.
    """
    # check to make sure id exists
    bertopic_sentence_transformer_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get(
        db, request.sentence_transformer_id)

    validate_obj(bertopic_sentence_transformer_obj)

    # verify enough documents to actually handle clustering
    if len(request.document_ids) < MIN_BERTOPIC_DOCUMENTS:
        raise HTTPException(
            status_code=400, detail="must have at least 7 documents to find topics")

    bertopic_weak_learner_obj = None
    if request.weak_learner_id:
        # check to make sure id exists
        bertopic_weak_learner_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get(
            db, request.weak_learner_id)

        validate_obj(bertopic_weak_learner_obj)

    llm_pretrained_obj = None
    if request.summarization_model_id:
        # check to make sure id exists
        llm_pretrained_obj: LlmPretrainedModel = crud_llm_pretrained.llm_pretrained.get(
            db, request.summarization_model_id)

        validate_obj(llm_pretrained_obj)

    # get the documents
    documents = []
    for document_id in request.document_ids:
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
            if embedding_computation.bertopic_embedding_pretrained_id == request.sentence_transformer_id:
                next_value = embedding_computation.embedding_vector
                break

        precalculated_embeddings.append(next_value)

    document_df = crud_mattermost.mattermost_documents.get_document_dataframe(db, document_uuids=request.document_ids)

    # train the model
    basic_inference = BasicInference(bertopic_sentence_transformer_obj,
                                     s3,
                                     request.prompt_template,
                                     request.refine_template,
                                     bertopic_weak_learner_obj,
                                     llm_pretrained_obj,
                                     stop_word_list=request.stop_words)
    inference_output = basic_inference.train_bertopic_on_documents(db,
                                                                   documents, precalculated_embeddings=precalculated_embeddings, num_topics=request.num_topics,
                                                                   document_df=document_df,
                                                                   seed_topic_list=request.seed_topics,
                                                                   trends_only=request.trends_only,
                                                                   trend_depth=request.trend_depth)

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

    # upload the trained model to minio
    upload_success = pickle_and_upload_object_to_minio(
        object=inference_output.topic_model, id=new_bertopic_trained_obj.id, s3=s3)

    # if upload was successful, set the uploaded flag to true in the database using crud.bertopic_trained.update
    if upload_success:
        new_bertopic_trained_obj = crud.bertopic_trained.update(
            db, db_obj=new_bertopic_trained_obj, obj_in=BertopicTrainedUpdate(uploaded=True))

    # save the join table between the documents and the trained model
    # see here: https://docs.sqlalchemy.org/en/20/orm/basic_relationships.html#many-to-many
    # and here: https://stackoverflow.com/questions/25668092/flask-sqlalchemy-many-to-many-insert-data
    new_bertopic_trained_obj.trained_on_documents.extend(documents)
    db.add(new_bertopic_trained_obj)
    db.commit()

    # refresh the new trained model object model
    # (see docs here for info: https://docs.sqlalchemy.org/en/20/orm/session_state_management.html#refreshing-expiring)
    db.refresh(new_bertopic_trained_obj)

    # upload model-level visualizations
    visualize_model_words = BertopicVisualizationCreate(
        model_or_topic_id=new_bertopic_trained_obj.id,
        visualization_type=BertopicVisualizationTypeEnum.MODEL_WORDS,
        html_string=inference_output.model_word_visualization.to_html(),
        json_string=inference_output.model_word_visualization.to_json()
    )
    crud.bertopic_visualization.create(db, obj_in=visualize_model_words)
    visualize_model_clusters = BertopicVisualizationCreate(
        model_or_topic_id=new_bertopic_trained_obj.id,
        visualization_type=BertopicVisualizationTypeEnum.MODEL_CLUSTERS,
        html_string=inference_output.model_cluster_visualization.to_html(),
        json_string=inference_output.model_cluster_visualization.to_json()
    )
    crud.bertopic_visualization.create(db, obj_in=visualize_model_clusters)
    visualize_model_timeline = BertopicVisualizationCreate(
        model_or_topic_id=new_bertopic_trained_obj.id,
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
                                  obj_in=TopicSummaryUpdate(model_id=new_bertopic_trained_obj.id))

        visualize_topic_timeline = BertopicVisualizationCreate(
            model_or_topic_id=inference_output.topics[tid].id,
            visualization_type=BertopicVisualizationTypeEnum.TOPIC_TIMELINE,
            html_string=inference_output.topic_timeline_visualization[tid].to_html(),
            json_string=inference_output.topic_timeline_visualization[tid].to_json()
        )
        crud.bertopic_visualization.create(db, obj_in=visualize_topic_timeline)

    return new_bertopic_trained_obj


def validate_obj(obj: Union[BertopicEmbeddingPretrainedModel, None]):
    if not obj:
        raise HTTPException(
            status_code=422, detail=f"Invalid {str(obj.model_type)} id")

    if not obj.uploaded:
        raise HTTPException(
            status_code=422, detail=f"{str(obj.model_type)} pretrained model not uploaded")
