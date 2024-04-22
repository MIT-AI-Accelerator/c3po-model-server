import os
from tqdm import tqdm
from typing import Union
from datetime import datetime
import numpy as np
import pandas as pd
from bertopic import BERTopic
import hdbscan
from pydantic import BaseModel, StrictFloat, StrictInt, StrictBool, validator
from minio import Minio
from plotly.graph_objs import Figure
from ppg.schemas.bertopic.topic import TopicSummaryCreate
from app.core.logging import logger
from app.core.minio import download_pickled_object_from_minio
from ..models.document import DocumentModel
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from ..models.topic import TopicSummaryModel
from ..crud import crud_topic
from .weak_learning import WeakLearner, get_vectorizer
from .topic_summarization import topic_summarizer, detect_trending_topics, DEFAULT_N_REPR_DOCS

BASE_CKPT_DIR = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "./data")

DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 10
DEFAULT_HDBSCAN_MIN_SAMPLES = 10
DEFAULT_BERTOPIC_VISUALIZATION_WORDS = 5
DEFAULT_BERTOPIC_TIME_BINS = 20

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
    topics: list[TopicSummaryModel]
    model_word_visualization: Figure
    model_cluster_visualization: Figure
    model_timeline_visualization: Figure
    topic_timeline_visualization: list[Figure]

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
    precalculated_embeddings: list[list[StrictFloat] | None] | None
    num_topics: StrictInt | None

    class Config:
        arbitrary_types_allowed = True


class BuildTopicModelInputs(BaseModel):
    documents_text_list: list[str]
    document_timestamps: list[datetime]
    embeddings: np.ndarray
    num_topics: StrictInt
    seed_topic_list: list[list]

    @validator('documents_text_list')
    def documents_text_list_must_be_large_enough_for_inference(cls, v):
        # pylint: disable=no-self-argument
        if len(v) < 7:
            raise ValueError(
                'documents_text_list must have at least 7 documents')
        return v

    # ensure embeddings is same length as documents_text_list
    @validator('embeddings')
    def embeddings_must_be_same_length_as_documents_text_list(cls, v, values):
        # pylint: disable=no-self-argument
        if values.get('documents_text_list') and v.shape[0] != len(values['documents_text_list']):
            raise ValueError(
                'embeddings must be same length as documents_text_list')
        return v

    @validator('num_topics')
    def num_topics_at_least_two(cls, v):
        # pylint: disable=no-self-argument
        if v < 2:
            raise ValueError(
                'num_topics must be at least 2')
        return v

    class Config:
        arbitrary_types_allowed = True


class TopicDocumentData(BaseModel):
    document_text_list: list[str]
    document_timestamps: list[datetime]
    document_users: list[str]
    document_nicknames: list[str]
    document_links: list[str]
    embeddings: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class BasicInference:

    def __init__(self, sentence_transformer_obj, s3, prompt_template, refine_template, weak_learner_obj=None, topic_summarizer_obj=None, stop_word_list=[]):

        # validate input
        InitInputs(
            embedding_pretrained_model_obj=sentence_transformer_obj, s3=s3
        )

        # TODO: load from minio--HTTPException gets thrown if not there
        # would be a server error since the db object should say if it's uploaded or not
        self.sentence_model = download_pickled_object_from_minio(
            id=sentence_transformer_obj.id, s3=s3)

        self.sentence_transformer_obj = sentence_transformer_obj

        self.vectorizer = get_vectorizer(stop_word_list=stop_word_list)

        self.weak_learner_obj = weak_learner_obj
        self.label_model = None
        if weak_learner_obj:
            weak_models = download_pickled_object_from_minio(
                id=weak_learner_obj.id, s3=s3)
            self.vectorizer = weak_models[0]
            self.svm = weak_models[1]
            self.mlp = weak_models[2]
            self.label_model = weak_models[3]
            self.weak_learner = WeakLearner(
                self.vectorizer, self.svm, self.mlp, self.label_model)
            labeling_functions, self.label_applier = self.weak_learner.create_label_applier()

        self.topic_summarizer = None
        if topic_summarizer_obj:
            if not topic_summarizer.check_parameters(topic_summarizer_obj.id, prompt_template, refine_template):
                topic_summarizer.initialize_llm(
                    s3, topic_summarizer_obj, prompt_template, refine_template)
            self.topic_summarizer = topic_summarizer

    def get_document_info(self, topic_model, topic_document_data: TopicDocumentData, num_documents=DEFAULT_N_REPR_DOCS):

        # increase number of representative documents (BERTopic default is 3)
        document_info = topic_model.get_document_info(topic_document_data.document_text_list)
        repr_docs, _, _ = topic_model._extract_representative_docs(topic_model.c_tf_idf_,
                                                                   document_info,
                                                                   topic_model.topic_representations_,
                                                                   nr_samples=500,
                                                                   nr_repr_docs=num_documents)
        topic_model.representative_docs_ = repr_docs

        document_info = topic_model.get_document_info(topic_document_data.document_text_list)
        document_info['Timestamp'] = topic_document_data.document_timestamps
        document_info['User'] = topic_document_data.document_users
        document_info['Nickname'] = topic_document_data.document_nicknames
        document_info['Link'] = topic_document_data.document_links
        return document_info

    def train_bertopic_on_documents(self, db, documents, precalculated_embeddings, num_topics, document_df, seed_topic_list=None, num_related_docs=DEFAULT_N_REPR_DOCS, trends_only=False) -> BasicInferenceOutputs:
        # validate input
        TrainBertopicOnDocumentsInput(
            documents=documents, precalculated_embeddings=precalculated_embeddings, num_topics=num_topics)

        document_text_list = list(document_df['message'].values)
        (embeddings, updated_document_indicies) = self.calculate_document_embeddings(
            document_text_list, precalculated_embeddings)

        document_df.user_name.fillna(value='', inplace=True)
        document_df.nickname.fillna(value='', inplace=True)
        document_df.mm_link.fillna(value='', inplace=True)
        topic_document_data = TopicDocumentData(document_text_list = document_text_list,
                                                document_timestamps = list(document_df['create_at'].values),
                                                document_users = list(document_df['user_name'].values),
                                                document_nicknames = list(document_df['nickname'].values),
                                                document_links = list(document_df['mm_link'].values),
                                                embeddings = embeddings)

        (topic_model, filtered_topic_document_data) = self.build_topic_model(
            topic_document_data, num_topics, seed_topic_list)

        # per topic documents and summary
        document_info = self.get_document_info(
            topic_model,
            filtered_topic_document_data,
            num_related_docs)

        topics_over_time = topic_model.topics_over_time(
            filtered_topic_document_data.document_text_list,
            filtered_topic_document_data.document_timestamps,
            nr_bins=DEFAULT_BERTOPIC_TIME_BINS)

        topic_info, new_topic_obj_list, topic_timeline_visualization_list = self.create_topic_visualizations(
            document_info, topic_model, topics_over_time, num_related_docs, num_topics, trends_only)

        if not new_topic_obj_list:
            logger.warn('no topics found')

        topic_objs = crud_topic.topic_summary.create_all_using_id(
            db, obj_in_list=new_topic_obj_list)

        # model-level topic cluster visualization
        model_cluster_visualization = topic_model.visualize_documents(
            filtered_topic_document_data.document_text_list,
            embeddings=filtered_topic_document_data.embeddings,
            title='Topic Analysis')

        # visualize_barchart will error if only default cluster (topic id -1) is available
        model_word_visualization = Figure()
        model_timeline_visualization = Figure()
        if len(topic_info) > 1:
            # model-level topic word visualization
            model_word_visualization = topic_model.visualize_barchart(
                top_n_topics=num_topics, n_words=DEFAULT_BERTOPIC_VISUALIZATION_WORDS,
                title='Topic Word Scores')

            # model-level topic timeline visualization
            model_timeline_visualization = topic_model.visualize_topics_over_time(
                topics_over_time,
                topics=topic_info['Topic'].values,
                title='Topics Over Time',
                top_n_topics=num_topics
            )

        return BasicInferenceOutputs(
            documents=documents,
            embeddings=embeddings.tolist(),
            updated_document_indicies=updated_document_indicies,
            topic_model=topic_model,
            topics=topic_objs,
            model_word_visualization=model_word_visualization,
            model_cluster_visualization=model_cluster_visualization,
            model_timeline_visualization=model_timeline_visualization,
            topic_timeline_visualization=topic_timeline_visualization_list
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

    def build_topic_model(self, topic_document_data: TopicDocumentData, num_topics, seed_topic_list) -> BERTopic:
        # validate input
        BuildTopicModelInputs(
            documents_text_list=topic_document_data.document_text_list,
            document_timestamps=topic_document_data.document_timestamps,
            embeddings=topic_document_data.embeddings,
            num_topics=num_topics,
            seed_topic_list=seed_topic_list)

        if self.weak_learner_obj:
            data_test = pd.DataFrame(
                {'message': topic_document_data.document_text_list,
                 'timestamp': topic_document_data.document_timestamps,
                 'user': topic_document_data.document_users,
                 'nickname': topic_document_data.document_nicknames,
                 'link': topic_document_data.document_links})
            l_test = self.label_applier.apply(
                pd.DataFrame(data_test['message']))
            data_test['y_pred'] = self.label_model.predict(l_test)
            topic_document_data.embeddings = topic_document_data.embeddings[
                data_test['y_pred'] < 2]
            data_test = data_test[data_test['y_pred'] < 2]
            topic_document_data.document_text_list = list(data_test['message'])
            topic_document_data.document_timestamps = list(data_test['timestamp'])
            topic_document_data.document_users = list(data_test['user'])
            topic_document_data.document_nicknames = list(data_test['nickname'])
            topic_document_data.document_links = list(data_test['link'])

        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
                                        min_samples=DEFAULT_HDBSCAN_MIN_SAMPLES,
                                        metric='euclidean', prediction_data=True)
        topic_model = BERTopic(nr_topics=num_topics, seed_topic_list=seed_topic_list,
                               hdbscan_model=hdbscan_model, vectorizer_model=self.vectorizer)
        topic_model = topic_model.fit(topic_document_data.document_text_list,
                                      topic_document_data.embeddings)

        return (topic_model, topic_document_data)

    def create_topic_visualizations(self, document_info, topic_model, topics_over_time, num_related_docs, num_topics, trends_only):

        topic_info = topic_model.get_topic_info()
        trending_topic_ids = detect_trending_topics(document_info, topic_info)

        new_topic_obj_list = []
        topic_timeline_visualization_list = []
        for key, row in tqdm(topic_info.iterrows()):
            if row['Topic'] < 0:
                continue

            is_trending = False
            if any(tid == row['Topic'] for tid in trending_topic_ids):
                is_trending = True

            if not trends_only or is_trending:

                topic_docs = document_info[document_info.Representative_document][document_info['Topic'] ==
                    row['Topic']].sort_values('Probability', ascending=False).head(num_related_docs).reset_index()

                summary_text = 'topic summarization disabled'
                if self.topic_summarizer:

                    # saves only the summary (but output includes intermediate steps of how we get to the summary
                    # if we want to save that in the future for XAI or other reason e.g., output_summary['intermediate_steps'])
                    output_summary = self.topic_summarizer.get_summary(
                        topic_docs['Document'].to_list())
                    if output_summary:
                        summary_text = output_summary['output_text']

                # topic-level timeline visualization
                topic_timeline_visualization_list = topic_timeline_visualization_list + [topic_model.visualize_topics_over_time(
                    topics_over_time, topics=[row['Topic']], title='Topic Over Time: ' + row['Name'],
                    top_n_topics=num_topics)]

                new_topic_obj_list = new_topic_obj_list + [TopicSummaryCreate(
                    topic_id=row['Topic'],
                    name=row['Name'],
                    top_n_words=topic_docs['Top_n_words'].unique()[0],
                    top_n_documents=topic_docs[[
                        'Document', 'Timestamp', 'User', 'Nickname', 'Link', 'Probability']].to_dict(),
                    summary=summary_text,
                    is_trending=is_trending)]

            else:
                logger.info('skipping topic %d (not trending)' % row['Topic'])

        return topic_info, new_topic_obj_list, topic_timeline_visualization_list
