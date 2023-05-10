from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter

from app.aimodels.bertopic.schemas.document_embedding_computation import DocumentEmbeddingComputationCreate
from ..ai_services.basic_inference import BasicInference
from app.dependencies import get_db
from ..schemas.bertopic_trained import BertopicTrained
from sqlalchemy.orm import Session
from .. import crud
from ..models.bertopic_trained import BertopicTrainedModel
from ..schemas.bertopic_trained import BertopicTrainedCreate
from app.core.errors import ValidationError, HTTPValidationError
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel

router = APIRouter(
    prefix=""
)


class TrainModelRequest(BaseModel):
    bertopic_embedding_pretrained_id: UUID4
    document_ids: list[UUID4] = []


@router.post(
    "/train",
    response_model=Union[BertopicTrained, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="Train BERTopic on text",
    response_description="Trained Model and Plotly Visualization config"
)
def train_bertopic_post(request: TrainModelRequest, db: Session = Depends(get_db)) -> (
    Union[BertopicTrained, HTTPValidationError]
):
    """
    Train a BERTopic model on text.
    """
    # check to make sure id exists
    bertopic_embedding_pretrained_obj: BertopicEmbeddingPretrainedModel = crud.bertopic_embedding_pretrained.get(
        db, request.bertopic_embedding_pretrained_id)
    if not bertopic_embedding_pretrained_obj:
        return HTTPValidationError(detail=[ValidationError(loc=['path', 'bertopic model upload'], msg='Invalid pretrained model id', type='value_error')])

    # check to make sure bertopic_obj has and embedding layer
    if not bertopic_embedding_pretrained_obj.uploaded:
        return HTTPValidationError(detail=[ValidationError(loc=['path', 'bertopic_id'], msg='BERTopic model has no embedding layer', type='value_error')])

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
            if embedding_computation.bertopic_embedding_pretrained_id == request.bertopic_embedding_pretrained_id:
                next_value = embedding_computation.embedding_vector
                break

        precalculated_embeddings.append(next_value)

    # train the model
    basic_inference = BasicInference(bertopic_embedding_pretrained_obj)
    inference_output = basic_inference.train_bertopic_on_documents(
        documents, precalculated_embeddings=precalculated_embeddings, num_topics=1)
    new_plotly_bubble_config = inference_output.plotly_bubble_config

    # save calculated embeddings computations
    new_embedding_computation_obj_list = [DocumentEmbeddingComputationCreate(
        document_id=documents[i].id,
        bertopic_embedding_pretrained_id=request.bertopic_embedding_pretrained_id,
        embedding_vector=inference_output.embeddings[i]
    ) for i, wasUpdated in enumerate(inference_output.updated_document_indicies) if wasUpdated]

    crud.document_embedding_computation.create_all_using_id(
        db, obj_in_list=new_embedding_computation_obj_list)

    # refresh the documents
    documents = crud.document.refresh_all_by_id(
        db, db_obj_ids=request.document_ids)

    # upload the trained model to minio
    # save inference_output.topic_model
    upload_success = True

    # create and save a trained model object
    bertopic_trained_obj = BertopicTrainedCreate(
        plotly_bubble_config=new_plotly_bubble_config,
        uploaded=upload_success
    )

    new_bertopic_trained_obj: BertopicTrainedModel = crud.bertopic_trained.create_with_embedding_pretrained_id(
        db, obj_in=bertopic_trained_obj, embedding_pretrained_id=request.bertopic_embedding_pretrained_id)

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
