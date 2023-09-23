from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from app.aimodels.bertopic.schemas.document_embedding_computation import DocumentEmbeddingComputationCreate
from minio import Minio
from app.core.minio import pickle_and_upload_object_to_minio
from ..ai_services.basic_inference import BasicInference
from app.dependencies import get_db, get_minio
from sqlalchemy.orm import Session
from .. import crud
from ..models.bertopic_trained import BertopicTrainedModel
from ..schemas.bertopic_trained import BertopicTrained, BertopicTrainedCreate, BertopicTrainedUpdate
from ..schemas.topic import TopicSummary, TopicSummaryUpdate
from ..ai_services.topic_summarization import MAP_PROMPT_TEMPLATE, COMBINE_PROMPT_TEMPLATE
from app.core.errors import ValidationError, HTTPValidationError
from app.core.config import settings
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from app.aimodels.gpt4all.models import Gpt4AllPretrainedModel
from app.aimodels.gpt4all.crud import crud_gpt4all_pretrained

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
    map_prompt_template: str = MAP_PROMPT_TEMPLATE
    combine_prompt_template: str = COMBINE_PROMPT_TEMPLATE


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
    if len(request.document_ids) < 7:
        raise HTTPException(
            status_code=400, detail="must have at least 7 documents to find topics")

    bertopic_weak_learner_obj = None
    if request.weak_learner_id:
        # check to make sure id exists
        bertopic_weak_learner_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get(
            db, request.weak_learner_id)

        validate_obj(bertopic_weak_learner_obj)

    gpt4all_pretrained_obj = None
    if request.summarization_model_id:
        # check to make sure id exists
        gpt4all_pretrained_obj: Gpt4AllPretrainedModel = crud_gpt4all_pretrained.gpt4all_pretrained.get(
            db, request.summarization_model_id)

        validate_obj(gpt4all_pretrained_obj)

    # get the documents
    documents = []
    for document_id in request.document_ids:
        document_obj = crud.document.get(db, document_id)
        if not document_obj:
            return HTTPValidationError(detail=[ValidationError(loc=['path', 'document_id'], msg=f'Invalid document id {document_id}', type='value_error')])

        documents.append(document_obj)

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

    # train the model
    basic_inference = BasicInference(
        bertopic_sentence_transformer_obj, s3, request.map_prompt_template, request.combine_prompt_template, bertopic_weak_learner_obj, gpt4all_pretrained_obj)
    inference_output = basic_inference.train_bertopic_on_documents(db,
                                                                   documents, precalculated_embeddings=precalculated_embeddings, num_topics=request.num_topics,
                                                                   seed_topic_list=request.seed_topics)

    new_topic_cluster_visualization = inference_output.topic_cluster_visualization
    new_topic_word_visualization = inference_output.topic_word_visualization

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
        map_prompt_template=request.map_prompt_template,
        combine_prompt_template=request.combine_prompt_template,
        topic_word_visualization=new_topic_word_visualization,
        topic_cluster_visualization=new_topic_cluster_visualization,
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

    [crud.topic_summary.update(db, db_obj=topic, obj_in=TopicSummaryUpdate(
        model_id=new_bertopic_trained_obj.id)) for topic in inference_output.topics]

    return new_bertopic_trained_obj


def validate_obj(obj: Union[BertopicEmbeddingPretrainedModel, None]):
    if not obj:
        raise HTTPException(
            status_code=422, detail=f"Invalid {str(obj.model_type)} id")

    if not obj.uploaded:
        raise HTTPException(
            status_code=422, detail=f"{str(obj.model_type)} pretrained model not uploaded")


@router.get(
    "/model/{id}/visualize_topic_clusters",
    response_class=HTMLResponse,
    summary="Retrieve a BERTopic document cluster visualization",
    response_description="Retrieved a BERTopic document cluster visualization")
async def visualize_topic_clusters(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic document cluster visualization

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """

    model_obj = crud.bertopic_trained.get(db, id)
    if not model_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic trained model not found")

    return model_obj.topic_cluster_visualization


@router.get(
    "/model/{id}/visualize_topic_words",
    response_class=HTMLResponse,
    summary="Retrieve a BERTopic word probability visualization",
    response_description="Retrieved a BERTopic word probability visualization")
async def visualize_topic_words(id: UUID4, db: Session = Depends(get_db)):
    """
    Retrieve a BERTopic word probability visualization

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """

    model_obj = crud.bertopic_trained.get(db, id)
    if not model_obj:
        raise HTTPException(
            status_code=422, detail=f"BERTopic trained model not found")

    return model_obj.topic_word_visualization


@router.get(
    "/model/{id}/topics",
    response_model=Union[list[TopicSummary], HTTPValidationError],
    summary="Retrieve topics for a trained BERTopic model",
    response_description="Retrieved topics for a trained BERTopic model")
async def get_model_topics(id: UUID4, db: Session = Depends(get_db)) -> (
    Union[list[TopicSummary], HTTPValidationError]
):
    """
    Retrieve topics for a trained BERTopic model

    - **trained_model_id**: Required.  Trained BERTopic model ID.
    """

    if not crud.bertopic_trained.get(db, id):
        raise HTTPException(
            status_code=422, detail=f"BERTopic trained model not found")

    return crud.topic_summary.get_by_model_id(db, model_id=id)
