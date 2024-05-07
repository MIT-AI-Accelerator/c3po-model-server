import uuid
import json
import typing
import pytest
import pandas as pd
from pytest_mock import MockerFixture
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import versioned_app
from app.aimodels.bertopic.ai_services.topic_summarization import TopicSummarizer, detect_trending_topics, \
    detect_trending_topics_single_day, DEFAULT_PROMPT_TEMPLATE, DEFAULT_REFINE_TEMPLATE, DEFAULT_LLM_TEMP

client = TestClient(versioned_app)

@pytest.fixture(scope='module')
def mock_model_obj():
    mock_model_obj = MagicMock()
    mock_model_obj.model_type = 'mistral'
    mock_model_obj.id = str(uuid.uuid4())
    return mock_model_obj


@pytest.fixture(scope='module')
def mock_llm():
    return MagicMock()


@pytest.fixture(scope='module')
def document_info_train() -> typing.Callable[[int, int, int], pd.DataFrame]:
    def _document_info_train(num_docs: int, topic_id_one: int, topic_id_two: int) -> pd.DataFrame:
        # posts per day for each topic id will always have mean 1 and std dev 0
        documents = ['train_blah_' + str(i) for i in range(num_docs)]
        today = datetime.today()
        timestamps = [today - timedelta(days=i) for i in range(num_docs)]
        topics_one = [topic_id_one] * (num_docs // 2)
        topics_two = [topic_id_two for _ in range(num_docs - len(topics_one))]
        topics = topics_one + topics_two

        return pd.DataFrame({'Document': documents,
                             'Timestamp': timestamps,
                             'Topic': topics})

    return _document_info_train


# test llm requires initialization
def test_initialize_llm_invalid():
    topic_summarizer = TopicSummarizer()
    assert not topic_summarizer.check_parameters(uuid.uuid4(), DEFAULT_PROMPT_TEMPLATE, DEFAULT_REFINE_TEMPLATE)
    assert topic_summarizer.get_summary(['document']) == 'topic summarization disabled'


@pytest.mark.parametrize('isfile_side_effect', [
    (True, False),
    (False, False),
    (False, True)
])
def test_initialize_llm(mock_s3: MagicMock,
                        mock_model_obj: MagicMock,
                        mock_llm: MagicMock,
                        mocker: MockerFixture,
                        isfile_side_effect: tuple[bool, bool]):

    # mock external classes and functions we don't want to test
    mocker.patch('os.path.isfile', side_effect=isfile_side_effect)  # conditional paths determined here

    # don't create any files
    mocker.patch(
        'app.aimodels.bertopic.ai_services.topic_summarization.Path', return_value=MagicMock(), autospec=True)
    mocker.patch(
        'app.aimodels.bertopic.ai_services.topic_summarization.download_file_from_minio', return_value=None)  # don't touch minio
    mocker.patch('app.aimodels.bertopic.ai_services.topic_summarization.CTransformers', return_value=mock_llm)

    ts = TopicSummarizer()
    ts.initialize_llm(mock_s3,
                      mock_model_obj,
                      DEFAULT_PROMPT_TEMPLATE,
                      DEFAULT_REFINE_TEMPLATE,
                      temp=DEFAULT_LLM_TEMP)

    assert ts.model_type == mock_model_obj.model_type
    assert ts.model_id == mock_model_obj.id
    assert ts.llm == mock_llm
    assert ts.prompt_template == DEFAULT_PROMPT_TEMPLATE
    assert ts.refine_template == DEFAULT_REFINE_TEMPLATE
    assert ts.temp == DEFAULT_LLM_TEMP


# test acronym expansion
def test_fix_text():
    acronym_dictionary = dict({'PPG': 'Prototype Proving Ground (PPG)'})
    client.post("/v1/upload_acronym_dictionary", params={'acronym_dictionary': json.dumps(acronym_dictionary)})

    documents = ['this is a test of the PPG.',
                 'this is a test of the PPG?',
                 'this is a test of the PPG!',
                 '']
    documents_no_punc = ['this is a test of the PPG']

    fixed_documents = TopicSummarizer().fix_text(documents)
    fixed_documents_no_punc = TopicSummarizer().fix_text(documents_no_punc)

    for fixed_doc, doc in zip(fixed_documents, documents):
        assert fixed_doc == doc.replace('PPG', acronym_dictionary['PPG'])
    assert fixed_documents_no_punc[0] == documents_no_punc[0].replace('PPG', acronym_dictionary['PPG']) + '.'


@pytest.mark.parametrize('num_docs,num_trending_day,num_other_days,trend_depth,expected', [
    (15, 10, 5, 30, True),
    (10, 4, 6, 30, False),  # not enough posts
    (15, 10, 5, 0, False),  # no trend depth
])
def test_detect_trending_topics(num_docs, num_trending_day, num_other_days, trend_depth, expected):
    document_text_list = ['blah_' + str(i) for i in range(num_docs)]

    trending_topic_id = 0
    today = datetime.today()
    trending_day_document_timestamps = [today for _ in range(num_trending_day)]
    other_days_document_timestamps = [today - timedelta(days=i+1) for i in range(num_other_days)]

    topics = [trending_topic_id] * num_docs
    document_timestamps = trending_day_document_timestamps + other_days_document_timestamps

    document_df = pd.DataFrame({'Document': document_text_list,
                               'Timestamp': document_timestamps,
                                'Topic': topics})
    result = detect_trending_topics(document_df, document_df, trend_depth=trend_depth)
    assert (trending_topic_id in result) == expected


def test_detect_trending_topics_trend_depth_one(mocker: MockerFixture):
    return_value = 'single day'
    mocker.patch(
        'app.aimodels.bertopic.ai_services.topic_summarization.detect_trending_topics_single_day',
        return_value=return_value)

    assert detect_trending_topics(pd.DataFrame(), pd.DataFrame(), trend_depth=1) == return_value


@pytest.mark.parametrize('num_docs,num_docs_one,num_docs_two,expected_one,expected_two', [
    (20, 10, 0, True, False),
    (20, 5, 5, True, True),
    (20, 4, 4, False, False),
    (20, 0, 0, False, False),
    (20, 0, 10, False, True)
])
def test_detect_trending_topics_single_day(document_info_train,
                                           num_docs,
                                           num_docs_one,
                                           num_docs_two,
                                           expected_one,
                                           expected_two):
    num_remaining = num_docs - num_docs_one - num_docs_two

    # build documents
    documents = ['test_blah_' + str(i) for i in range(num_docs)]

    # build timestamps
    today = datetime.today()
    yday = today - timedelta(days=1)
    ts_one = [today] * num_docs_one
    ts_two = [yday] * num_docs_two
    ts_lst = ts_one + ts_two + [yday - timedelta(days=i+1) for i in range(num_remaining)]

    # build topics
    trending_id_one, trending_id_two = 0, 1
    topics_one = [trending_id_one] * num_docs_one
    topics_two = [trending_id_two] * num_docs_two
    topics = topics_one + topics_two + [i + 1 + trending_id_two for i in range(num_remaining)]

    document_df_test = pd.DataFrame({'Document': documents,
                                     'Timestamp': ts_lst,
                                     'Topic': topics})
    doc_info_train = document_info_train(num_docs, trending_id_one, trending_id_two)

    trending_topic_ids = detect_trending_topics_single_day(doc_info_train, document_df_test)

    assert (trending_id_one in trending_topic_ids) == expected_one
    assert (trending_id_two in trending_topic_ids) == expected_two
