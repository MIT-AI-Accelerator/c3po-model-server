import os
from tqdm import tqdm
from typing import Union
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import hdbscan
from pydantic import BaseModel, StrictFloat, StrictInt, StrictBool, validator
from minio import Minio
from app.core.minio import download_pickled_object_from_minio
from ..models.document import DocumentModel
from ..models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel
from ..models.topic import TopicSummaryModel
from ..schemas.topic import TopicSummaryCreate
from ..crud import crud_topic
from .weak_learning import WeakLearner
from .topic_summarization import TopicSummarizer, topic_summarizer, DEFAULT_N_REPR_DOCS

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
    topic_word_visualization: str
    topic_cluster_visualization: str

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


class BasicInference:

    def __init__(self, sentence_transformer_obj, s3, map_prompt_template, combine_prompt_template, weak_learner_obj=None, topic_summarizer_obj=None):

        # validate input
        InitInputs(
            embedding_pretrained_model_obj=sentence_transformer_obj, s3=s3
        )

        # TODO: load from minio--HTTPException gets thrown if not there
        # would be a server error since the db object should say if it's uploaded or not
        self.sentence_model = download_pickled_object_from_minio(
            id=sentence_transformer_obj.id, s3=s3)

        self.sentence_transformer_obj = sentence_transformer_obj

        self.vectorizer = CountVectorizer(
            stop_words="english", ngram_range=(1, 2))

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
            if not topic_summarizer.check_parameters(topic_summarizer_obj.id, map_prompt_template, combine_prompt_template):
                topic_summarizer.initialize_llm(
                    s3, topic_summarizer_obj, map_prompt_template, combine_prompt_template)
            self.topic_summarizer = topic_summarizer

    def get_document_info(self, topic_model, documents, timestamps, num_documents=DEFAULT_N_REPR_DOCS):

        # increase number of representative documents (BERTopic default is 3)
        document_info = topic_model.get_document_info(documents)
        repr_docs, _, _ = topic_model._extract_representative_docs(topic_model.c_tf_idf_,
                                                                   document_info,
                                                                   topic_model.topic_representations_,
                                                                   nr_samples=500,
                                                                   nr_repr_docs=num_documents)
        topic_model.representative_docs_ = repr_docs

        document_info = topic_model.get_document_info(documents)
        document_info['Timestamp'] = timestamps
        return document_info

    def train_bertopic_on_documents(self, db, documents, precalculated_embeddings, num_topics, seed_topic_list=None, num_related_docs=DEFAULT_N_REPR_DOCS) -> BasicInferenceOutputs:
        # validate input
        TrainBertopicOnDocumentsInput(
            documents=documents, precalculated_embeddings=precalculated_embeddings, num_topics=num_topics)

        documents_text_list = [document.text for document in documents]
        document_timestamps = [
            document.original_created_time for document in documents]

        (embeddings, updated_document_indicies) = self.calculate_document_embeddings(
            documents_text_list, precalculated_embeddings)

        (topic_model, filtered_embeddings, filtered_documents_text_list, filtered_timestamps) = self.build_topic_model(
            documents_text_list, document_timestamps, embeddings, num_topics, seed_topic_list)

        # per topic documents and summary
        document_info = self.get_document_info(
            topic_model, filtered_documents_text_list, filtered_timestamps, num_related_docs)
        topics_over_time = topic_model.topics_over_time(
            filtered_documents_text_list, filtered_timestamps, nr_bins=DEFAULT_BERTOPIC_TIME_BINS)

        topic_info = topic_model.get_topic_info()
        new_topic_obj_list = []
        for key, row in tqdm(topic_info.iterrows()):
            if row['Topic'] < 0:
                continue

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

            topic_timeline_visualization = topic_model.visualize_topics_over_time(
                topics_over_time, topics=[row['Topic']], title='Topic Over Time: ' + row['Name'],
                top_n_topics=num_topics).to_html()

            new_topic_obj_list = new_topic_obj_list + [TopicSummaryCreate(
                topic_id=row['Topic'],
                name=row['Name'],
                top_n_words=topic_docs['Top_n_words'].unique()[0],
                top_n_documents=topic_docs[[
                    'Document', 'Timestamp', 'Probability']].to_dict(),
                summary=summary_text,
                topic_timeline_visualization=topic_timeline_visualization)]

        topic_objs = crud_topic.topic_summary.create_all_using_id(
            db, obj_in_list=new_topic_obj_list)

        # output topic cluster visualization as an html string
        topic_cluster_visualization = topic_model.visualize_documents(
            filtered_documents_text_list, embeddings=filtered_embeddings, title='Topic Analysis').to_html()

        # visualize_barchart will error if only default cluster (topic id -1) is available
        if len(topic_info) > 1:
            # output topic word visualization as an html string
            topic_word_visualization = topic_model.visualize_barchart(
                top_n_topics=num_topics, n_words=DEFAULT_BERTOPIC_VISUALIZATION_WORDS,
                title='Topic Word Scores').to_html()
        else:
            topic_word_visualization = "<html>No topics to display</html>"

        return BasicInferenceOutputs(
            documents=documents,
            embeddings=embeddings.tolist(),
            updated_document_indicies=updated_document_indicies,
            topic_model=topic_model,
            topics=topic_objs,
            topic_word_visualization=topic_word_visualization,
            topic_cluster_visualization=topic_cluster_visualization
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

    def build_topic_model(self, documents_text_list, document_timestamps, embeddings, num_topics, seed_topic_list) -> BERTopic:

        # validate input
        BuildTopicModelInputs(
            documents_text_list=documents_text_list, document_timestamps=document_timestamps, embeddings=embeddings, num_topics=num_topics, seed_topic_list=seed_topic_list)

        if self.weak_learner_obj:
            data_test = pd.DataFrame(
                {'message': documents_text_list, 'timestamp': document_timestamps})
            l_test = self.label_applier.apply(
                pd.DataFrame(data_test['message']))
            data_test['y_pred'] = self.label_model.predict(l_test)
            embeddings = embeddings[data_test['y_pred'] < 2]
            data_test = data_test[data_test['y_pred'] < 2]
            documents_text_list = list(data_test['message'])
            document_timestamps = list(data_test['timestamp'])

        # TODO convert message utterances to conversation threads
        # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=36313268
        # see convertThreads in aimodels_test_plus_summarization

        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
                                        min_samples=DEFAULT_HDBSCAN_MIN_SAMPLES,
                                        metric='euclidean', prediction_data=True)
        topic_model = BERTopic(nr_topics=num_topics, seed_topic_list=seed_topic_list,
                               hdbscan_model=hdbscan_model, vectorizer_model=self.vectorizer)
        topic_model = topic_model.fit(documents_text_list, embeddings)

        return (topic_model, embeddings, documents_text_list, document_timestamps)
