import os
import json
from typing import Union
from fastapi import HTTPException
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import hdbscan
from pydantic import BaseModel, StrictFloat, StrictInt, StrictBool, validator

from app.core.minio import download_pickled_object_from_minio
from minio import Minio

from ..models.document import DocumentModel
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from .weak_learning import WeakLearner

BASE_CKPT_DIR = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "./data")


# keep this approach or change to pydantic.dataclasses?
# see here: https://docs.pydantic.dev/usage/dataclasses/
# note that Config is done differently if using pydantic.dataclasses

class InitInputs(BaseModel):
    embedding_pretrained_model_obj: BertopicEmbeddingPretrainedModel
    s3: Minio

    # ensure that model type is defined
    @validator('embedding_pretrained_model_obj')
    def embedding_pretrained_model_obj_must_have_model_type_and_be_uploaded(cls, v):
        # pylint: disable=no-self-argument
        if not v.model_type:
            raise ValueError(
                'embedding_pretrained_model_obj must have model_type')
        if not v.uploaded:
            raise ValueError(
                'embedding_pretrained_model_obj must be uploaded')

        return v

    class Config:
        arbitrary_types_allowed = True


class BasicInferenceOutputs(BaseModel):
    documents: list[DocumentModel]
    embeddings: list[list[StrictFloat]]
    updated_document_indicies: list[StrictBool]
    topic_model: BERTopic
    plotly_bubble_config: dict

    # ensure that documents is same length as embeddings
    @validator('embeddings')
    def embeddings_must_be_same_length_as_documents(cls, v, values):
        # pylint: disable=no-self-argument
        if len(v) != len(values['documents']):
            raise ValueError(
                'embeddings must be same length as documents')
        return v

    # ensure that updated_document_indicies is same length as documents if given
    @validator('updated_document_indicies')
    def updated_document_indicies_must_be_same_length_as_documents(cls, v, values):
        # pylint: disable=no-self-argument
        if len(v) != len(values['documents']):
            raise ValueError(
                'updated_document_indicies must be same length as documents')
        return v

    class Config:
        arbitrary_types_allowed = True


class CalculateDocumentEmbeddingsInputs(BaseModel):
    documents_text_list: list[str]
    precalculated_embeddings: Union[list[Union[list[StrictFloat], None]], None] = None

    @validator('documents_text_list')
    def documents_text_list_must_be_non_empty(cls, v):
        # pylint: disable=no-self-argument
        if len(v) == 0:
            raise ValueError(
                'documents_text_list must be non-empty')
        return v

    # if given, ensure precalculated_embeddings is same length as documents_text_list
    @validator('precalculated_embeddings')
    def precalculated_embeddings_must_be_same_length_as_documents_text_list(cls, v, values):
        # pylint: disable=no-self-argument
        if v and len(v) != len(values['documents_text_list']):
            raise ValueError(
                'precalculated_embeddings must be same length as documents_text_list if given')
        return v

    class Config:
        arbitrary_types_allowed = True


class TrainBertopicOnDocumentsInput(BaseModel):
    documents: list[DocumentModel]
    precalculated_embeddings: Union[list[Union[list[StrictFloat], None]], None] = None
    num_topics: StrictInt = 1

    class Config:
        arbitrary_types_allowed = True


class BuildTopicModelInputs(BaseModel):
    documents_text_list: list[str]
    embeddings: np.ndarray
    num_topics: StrictInt = 1

    @validator('documents_text_list')
    def documents_text_list_must_be_large_enough_for_inference(cls, v):
        # pylint: disable=no-self-argument
        if len(v) < 10:
            raise ValueError(
                'documents_text_list must have at least 10 documents')
        return v

    # ensure embeddings is same length as documents_text_list
    @validator('embeddings')
    def embeddings_must_be_same_length_as_documents_text_list(cls, v, values):
        # pylint: disable=no-self-argument
        if v.shape[0] != len(values['documents_text_list']):
            raise ValueError(
                'embeddings must be same length as documents_text_list')
        return v

    class Config:
        arbitrary_types_allowed = True


class BasicInference:

    def __init__(self, sentence_transformer_obj, s3, weak_learner_obj = None):

        # validate input
        InitInputs(
            embedding_pretrained_model_obj=sentence_transformer_obj, s3=s3
        )

        # TODO: load from minio--HTTPException gets thrown if not there
        # would be a server error since the db object should say if it's uploaded or not
        self.sentence_model = download_pickled_object_from_minio(
            id=sentence_transformer_obj.id, s3=s3)

        self.sentence_transformer_obj = sentence_transformer_obj

        self.weak_learner_obj = weak_learner_obj
        self.label_model = None
        if weak_learner_obj:
            weak_models = download_pickled_object_from_minio(id=weak_learner_obj.id, s3=s3)
            self.vectorizer = weak_models[0]
            self.svm = weak_models[1]
            self.mlp = weak_models[2]
            self.label_model = weak_models[3]
            self.weak_learner = WeakLearner(self.vectorizer, self.svm, self.mlp, self.label_model)
            labeling_functions, self.label_applier = self.weak_learner.create_label_applier()
        
    def train_bertopic_on_documents(self, documents, precalculated_embeddings, num_topics = 1, seed_topic_list = []) -> BasicInferenceOutputs:
        # validate input
        TrainBertopicOnDocumentsInput(
            documents=documents, precalculated_embeddings=precalculated_embeddings, num_topics=num_topics)

        documents_text_list = [document.text for document in documents]

        (embeddings, updated_document_indicies) = self.calculate_document_embeddings(
            documents_text_list, precalculated_embeddings)

        topic_model = self.build_topic_model(
            documents_text_list, embeddings, num_topics, seed_topic_list)

        plotly_visualization = topic_model.visualize_documents(
            documents_text_list, embeddings=embeddings, title='Topic Analysis')

        # output plotly config to json string and convert to dict
        plotly_bubble_config = json.loads(plotly_visualization.to_json())

        return BasicInferenceOutputs(
            documents=documents,
            embeddings=embeddings.tolist(),
            updated_document_indicies=updated_document_indicies,
            topic_model=topic_model,
            plotly_bubble_config=plotly_bubble_config
        )

    def calculate_document_embeddings(
            self, documents_text_list, precalculated_embeddings) -> tuple[list[list[StrictFloat]], list[StrictBool]]:

        # validate input
        CalculateDocumentEmbeddingsInputs(
            documents_text_list=documents_text_list, precalculated_embeddings=precalculated_embeddings)

        if not precalculated_embeddings:
            return (self.sentence_model.encode(documents_text_list, show_progress_bar=True), [True] * len(documents_text_list))

        # only calculate embeddings on documents_text_list where the associated value in precalculated_embeddings is None
        new_documents_text_list = [documents_text_list[i]
                                   for i in range(len(documents_text_list)) if precalculated_embeddings[i] is None]

        new_embeddings = self.sentence_model.encode(
            new_documents_text_list, show_progress_bar=True).tolist()

        # combine new embeddings with precalculated embeddings
        embeddings = []
        updated_indices = []
        for i in range(len(documents_text_list)):
            if precalculated_embeddings[i] is not None:
                embeddings.append(precalculated_embeddings[i])
                updated_indices.append(False)
            else:
                embeddings.append(new_embeddings.pop(0))
                updated_indices.append(True)

        return (np.array(embeddings), updated_indices)

    def build_topic_model(self, documents_text_list, embeddings, num_topics, seed_topic_list) -> BERTopic:

        # validate input
        BuildTopicModelInputs(
            documents_text_list=documents_text_list, embeddings=embeddings, num_topics=num_topics)

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10,
                                        metric='euclidean', prediction_data=True)

        if self.weak_learner_obj:
            data_test = pd.DataFrame({'message': documents_text_list})
            l_test = self.label_applier.apply(pd.DataFrame(data_test['message']))
            y_pred = self.label_model.predict(l_test)
            test_messages = data_test['message'][y_pred<2]
            test_messages.index = list(range(test_messages.shape[0]))
            topic_model = BERTopic(nr_topics=num_topics, # seed_topic_list=seed_topic_list, TODO fix seed topic list
                                hdbscan_model=hdbscan_model, vectorizer_model=self.vectorizer,
                                embedding_model=self.sentence_model).fit(test_messages)
  
        else:
            topic_model = BERTopic(nr_topics=num_topics, seed_topic_list=seed_topic_list,
                                hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model).\
                                    fit(documents_text_list, embeddings)

        return topic_model
