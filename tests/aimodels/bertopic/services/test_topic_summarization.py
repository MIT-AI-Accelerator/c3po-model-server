import uuid
import json
from fastapi.testclient import TestClient
from app.main import versioned_app
from app.aimodels.bertopic.ai_services.topic_summarization import TopicSummarizer, MAP_PROMPT_TEMPLATE, COMBINE_PROMPT_TEMPLATE

client = TestClient(versioned_app)

# test llm requires initialization
def test_initalize_llm_invalid():
    topic_summarizer = TopicSummarizer()
    assert not topic_summarizer.check_parameters(uuid.uuid4(), MAP_PROMPT_TEMPLATE, COMBINE_PROMPT_TEMPLATE)
    assert topic_summarizer.get_summary(['document']) is None

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
