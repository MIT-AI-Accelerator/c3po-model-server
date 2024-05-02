import uuid
import json
import pytest
import pandas as pd
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from app.main import versioned_app
from app.aimodels.bertopic.ai_services.topic_summarization import TopicSummarizer, detect_trending_topics, \
    DEFAULT_PROMPT_TEMPLATE, DEFAULT_REFINE_TEMPLATE

client = TestClient(versioned_app)


# test llm requires initialization
def test_initalize_llm_invalid():
    topic_summarizer = TopicSummarizer()
    assert not topic_summarizer.check_parameters(uuid.uuid4(), DEFAULT_PROMPT_TEMPLATE, DEFAULT_REFINE_TEMPLATE)
    assert topic_summarizer.get_summary(['document']) == 'topic summarization disabled'


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

    trending_topic_id = str(uuid.uuid4())
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
