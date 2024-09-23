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
from umap import UMAP
from fastapi import HTTPException
from plotly.graph_objs import Figure
from ppg.schemas.bertopic.topic import TopicSummaryCreate
from app.core.logging import logger
from app.core.minio import download_pickled_object_from_minio
from app.core.config import get_label_dictionary
from ..models.document import DocumentModel
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from ..models.topic import TopicSummaryModel
from ..crud import crud_topic
from .weak_learning import WeakLearner, get_vectorizer
from .topic_summarization import topic_summarizer, detect_trending_topics, DEFAULT_N_REPR_DOCS, DEFAULT_TREND_DEPTH_DAYS

BASE_CKPT_DIR = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "./data")

MIN_BERTOPIC_DOCUMENTS = 7
DEFAULT_TRAIN_PERCENT = 0.7
DEFAULT_BERTOPIC_VISUALIZATION_WORDS = 5
DEFAULT_BERTOPIC_TIME_BINS = 20

DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 10
DEFAULT_HDBSCAN_MIN_SAMPLES = 10
DEFAULT_HDBSCAN_METRIC = 'euclidean'
DEFAULT_HDBSCAN_CLUSTER_METHOD = 'eom'

DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_COMPONENTS = 5
DEFAULT_UMAP_MIN_DIST = 0.0
DEFAULT_UMAP_METRIC = 'cosine'
DEFAULT_UMAP_RANDOM_STATE = 577


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
    document_messages: list[str]
    document_timestamps: list[datetime]
    document_users: list[str]
    document_nicknames: list[str]
    document_channels: list[str]
    document_links: list[str]
    document_metadata: list[dict]
    document_summarization_messages: list[str]
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
            if get_label_dictionary() != weak_models[4]:
                raise HTTPException(status_code=422, detail="Retrain weak learner (labeling_dict mismatch)")

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
        document_info['Message'] = topic_document_data.document_messages
        document_info['Timestamp'] = topic_document_data.document_timestamps
        document_info['User'] = topic_document_data.document_users
        document_info['Nickname'] = topic_document_data.document_nicknames
        document_info['Channel'] = topic_document_data.document_channels
        document_info['Link'] = topic_document_data.document_links
        document_info['Metadata'] = topic_document_data.document_metadata
        document_info['Summarization_Message'] = topic_document_data.document_summarization_messages
        return document_info

    def train_bertopic_on_documents(self, db, documents, precalculated_embeddings, num_topics, document_df, seed_topic_list=None, num_related_docs=DEFAULT_N_REPR_DOCS, trends_only=False, trend_depth=DEFAULT_TREND_DEPTH_DAYS, train_percent=DEFAULT_TRAIN_PERCENT) -> BasicInferenceOutputs:
        # validate input
        TrainBertopicOnDocumentsInput(
            documents=documents, precalculated_embeddings=precalculated_embeddings, num_topics=num_topics)

        document_text_list = list(document_df['message'].values)
        (embeddings, updated_document_indicies) = self.calculate_document_embeddings(
            document_text_list, precalculated_embeddings)

        document_df.user_name.fillna(value='', inplace=True)
        document_df.nickname.fillna(value='', inplace=True)
        document_df.channel_name.fillna(value='', inplace=True)
        document_df.mm_link.fillna(value='', inplace=True)
        topic_document_data = TopicDocumentData(document_text_list = document_text_list,
                                                document_messages = document_text_list,
                                                document_timestamps = list(document_df['create_at'].values),
                                                document_users = list(document_df['user_name'].values),
                                                document_nicknames = list(document_df['nickname'].values),
                                                document_channels = list(document_df['channel_name'].values),
                                                document_links = list(document_df['mm_link'].values),
                                                document_metadata = list(document_df['mm_metadata'].values),
                                                document_summarization_messages = list(document_df['summarization_message'].values),
                                                embeddings = embeddings)

        topic_model, topic_document_data_train, topic_document_data_test = self.build_topic_model(
            topic_document_data, num_topics, seed_topic_list, train_percent)

        # per topic documents and summary. note: this only works for the train documents
        document_info_train = self.get_document_info(
            topic_model,
            topic_document_data_train,
            num_related_docs)

        # update topics for test documents
        topics, probs = topic_model.transform(documents=topic_document_data_test.document_text_list,
                                              embeddings=topic_document_data_test.embeddings)

        # train documents needed for representative documents / summarization
        # test documents needed for trend detection
        document_df_test = pd.DataFrame({'Document': topic_document_data_test.document_text_list,
                                         'Message': topic_document_data_test.document_messages,
                                         'Timestamp': topic_document_data_test.document_timestamps,
                                         'Topic': topics})

        # note: removed topic_model.update_topics for topic_document_data_test - impacts topics_timeline and topic_words
        # adjust the following function if it is added back in
        topic_info, new_topic_obj_list, topics_over_time_test, topic_timeline_visualization_list = self.create_topic_visualizations(
            document_info_train, topic_model, document_df_test, num_related_docs, num_topics, trends_only, trend_depth)

        if not new_topic_obj_list:
            logger.warning('no topics found')

        topic_objs = crud_topic.topic_summary.create_all_using_id(
            db, obj_in_list=new_topic_obj_list)

        # model-level topic cluster visualization
        model_cluster_visualization = topic_model.visualize_documents(
            topic_document_data_train.document_text_list,
            embeddings=topic_document_data_train.embeddings,
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
                topics_over_time_test,
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

    def build_topic_model(self, topic_document_data: TopicDocumentData, num_topics, seed_topic_list, train_percent) -> BERTopic:
        # validate input
        BuildTopicModelInputs(
            documents_text_list = topic_document_data.document_text_list,
            document_timestamps = topic_document_data.document_timestamps,
            embeddings = topic_document_data.embeddings,
            num_topics = num_topics,
            seed_topic_list = seed_topic_list)

        data_train = pd.DataFrame(
            {'message': topic_document_data.document_text_list,
                'timestamp': topic_document_data.document_timestamps,
                'user': topic_document_data.document_users,
                'nickname': topic_document_data.document_nicknames,
                'channel': topic_document_data.document_channels,
                'link': topic_document_data.document_links,
                'metadata': topic_document_data.document_metadata,
                'summarization_message': topic_document_data.document_summarization_messages})

        if self.weak_learner_obj:
            l_test = self.label_applier.apply(
                pd.DataFrame(data_train['message']))
            data_train['y_pred'] = self.label_model.predict(l_test)
            topic_document_data.embeddings = topic_document_data.embeddings[
                data_train['y_pred'] < 2]
            data_train = data_train[data_train['y_pred'] < 2]

        # convert documents to lowercase prior to stopword removal
        data_train['document'] = data_train['message'].str.lower()

        # remove emojis from topic document text
        data_train['document'] = data_train['document'].replace(to_replace =':[a-zA-Z0-9_]*:', value = '', regex = True)

        # split data, train, then infer. assumes documents, embeddings
        # sorted by timestamp previously (in train request)
        train_len = round(len(data_train) * train_percent)
        data_test = data_train[train_len:]
        data_train = data_train[:train_len-1]

        if len(data_train) < MIN_BERTOPIC_DOCUMENTS or len(data_test) < MIN_BERTOPIC_DOCUMENTS:
            logger.error('document set reduced below minimum required for topic modeling')

        topic_document_data_train = TopicDocumentData(document_text_list = list(data_train['document']),
                                                      document_messages = list(data_train['message']),
                                                      document_timestamps = list(data_train['timestamp']),
                                                      document_users = list(data_train['user']),
                                                      document_nicknames = list(data_train['nickname']),
                                                      document_channels = list(data_train['channel']),
                                                      document_links = list(data_train['link']),
                                                      document_metadata = list(data_train['metadata']),
                                                      document_summarization_messages = list(data_train['summarization_message']),
                                                      embeddings = topic_document_data.embeddings[:train_len-1])
        topic_document_data_test = TopicDocumentData(document_text_list = list(data_test['document']),
                                                     document_messages = list(data_test['message']),
                                                     document_timestamps = list(data_test['timestamp']),
                                                     document_users = list(data_test['user']),
                                                     document_nicknames = list(data_test['nickname']),
                                                     document_channels= list(data_test['channel']),
                                                     document_links = list(data_test['link']),
                                                     document_metadata = list(data_test['metadata']),
                                                     document_summarization_messages = list(data_test['summarization_message']),
                                                     embeddings = topic_document_data.embeddings[train_len:])

        umap_model = UMAP(n_neighbors = DEFAULT_UMAP_NEIGHBORS,
                          n_components = DEFAULT_UMAP_COMPONENTS,
                          min_dist = DEFAULT_UMAP_MIN_DIST,
                          metric = DEFAULT_UMAP_METRIC,
                          random_state = DEFAULT_UMAP_RANDOM_STATE)
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
                                        min_samples = DEFAULT_HDBSCAN_MIN_SAMPLES,
                                        metric = DEFAULT_HDBSCAN_METRIC,
                                        cluster_selection_method = DEFAULT_HDBSCAN_CLUSTER_METHOD,
                                        prediction_data = True)
        topic_model = BERTopic(nr_topics = num_topics,
                               seed_topic_list = seed_topic_list,
                               hdbscan_model = hdbscan_model,
                               vectorizer_model = self.vectorizer,
                               umap_model = umap_model)
        topic_model = topic_model.fit(documents = topic_document_data_train.document_text_list,
                                      embeddings = topic_document_data_train.embeddings)

        return topic_model, topic_document_data_train, topic_document_data_test

    def create_topic_visualizations(self, document_info_train, topic_model, document_df_test, num_related_docs, num_topics, trends_only, trend_depth):

        topic_info = topic_model.get_topic_info()

        trending_topic_ids = detect_trending_topics(document_info_train, document_df_test, trend_depth)

        # if update_topics is run in train_bertopic_on_documents, use document_df_test
        # otherwise, use document_info_train
        topics_over_time = topic_model.topics_over_time(
            docs=document_info_train['Document'],
            timestamps=document_info_train['Timestamp'],
            nr_bins=DEFAULT_BERTOPIC_TIME_BINS)

        # set is_trending outside of loop, skip -1 cluster
        topic_info = topic_info[topic_info['Topic'] >= 0]
        topic_info['is_trending'] = topic_info['Topic'].isin(trending_topic_ids)

        new_topic_obj_list = []
        topic_timeline_visualization_list = []
        for key, row in tqdm(topic_info.iterrows()):

            if not trends_only or row['is_trending']:

                topic_docs = document_info_train[document_info_train.Representative_document][document_info_train['Topic'] ==
                    row['Topic']].sort_values('Probability', ascending=False).head(num_related_docs).reset_index()

                summary_text = 'topic summarization disabled'
                if self.topic_summarizer:
                    summary_text = self.topic_summarizer.get_summary(topic_docs['Summarization_Message'].to_list())

                # topic-level timeline visualization
                topic_timeline_visualization_list = topic_timeline_visualization_list + [topic_model.visualize_topics_over_time(
                    topics_over_time, topics=[row['Topic']], title='Topic Over Time: ' + row['Name'],
                    top_n_topics=num_topics)]

                new_topic_obj_list = new_topic_obj_list + [TopicSummaryCreate(
                    topic_id=row['Topic'],
                    name=row['Name'],
                    top_n_words=topic_docs['Top_n_words'].unique()[0],
                    top_n_documents=topic_docs.rename(columns={'Document': 'Lowercase', 'Summarization_Message': 'Document'})[[
                        'Document', 'Timestamp', 'User', 'Nickname', 'Channel', 'Link', 'Probability']].to_dict(),
                    summary=summary_text,
                    is_trending=row['is_trending'])]

            else:
                logger.info('skipping topic %d (not trending)' % row['Topic'])

        return topic_info, new_topic_obj_list, topics_over_time, topic_timeline_visualization_list
