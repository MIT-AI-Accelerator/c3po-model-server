import os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from string import punctuation
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from scipy.stats import shapiro
from app.core.logging import logger
from app.core.model_cache import MODEL_CACHE_BASEDIR
from app.core.s3 import download_file_from_s3
from app.core.config import get_acronym_dictionary

# default templates for topic summarization
DEFAULT_PROMPT_TEMPLATE = """Summarize this content:
    {text}
    SUMMARY:
    """
DEFAULT_REFINE_TEMPLATE = (
        "Your job is to produce a final summary\n"
        "Here's the existing summary: {existing_answer}\n "
        "Now add to it based on the following context (only if needed):\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "SUMMARY: "
    )

# default parameters for topic summarization
DEFAULT_N_REPR_DOCS = 5
DEFAULT_LLM_TEMP = 0.69
DEFAULT_MAX_NEW_TOKENS = 2000
DEFAULT_CONTEXT_LENGTH = 6000

# default parameters for trend detection
DEFAULT_TREND_DEPTH_DAYS = 7
DEFAULT_SHAPIRO_P_VALUE = 0.001
DEFAULT_POST_THRESHOLD = 5
DEFAULT_TRAIN_FACTOR = 3

class TopicSummarizer:

    def __init__(self):
        self.model_type = None
        self.model_id = None
        self.prompt_template = None
        self.refine_template = None
        self.temp = None
        self.top_p = None
        self.top_p = None
        self.llm = None

    def initialize_llm(self, s3, model_obj,
                       prompt_template=DEFAULT_PROMPT_TEMPLATE,
                       refine_template=DEFAULT_REFINE_TEMPLATE,
                       temp=DEFAULT_LLM_TEMP):

        self.model_type = model_obj.model_type
        self.model_id = model_obj.id
        llm_path = os.path.join(MODEL_CACHE_BASEDIR, self.model_type)

        # download gpt4all model binary
        if not os.path.isfile(llm_path):
            # Create the directory if it doesn't exist
            Path(llm_path).parent.mkdir(parents=True, exist_ok=True)

            # Download the file from S3
            logger.info(f"Downloading model from S3 to {llm_path}")
            download_file_from_s3(model_obj.id, s3, filename=llm_path)

            if not os.path.isfile(llm_path):
                logger.error(f"Error downloading model from S3 to {llm_path}")
            else:
                logger.info(f"Downloaded model from S3 to {llm_path}")

        config = {'max_new_tokens': DEFAULT_MAX_NEW_TOKENS,
                  'temperature': DEFAULT_LLM_TEMP,
                  'context_length': DEFAULT_CONTEXT_LENGTH}
        self.llm = CTransformers(
            model = llm_path,
            model_type = 'mistral',
            config = config,
            threads = os.cpu_count()
        )
        self.prompt_template = prompt_template
        self.refine_template = refine_template

        # TODO add configuration parameter for temp
        # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=36312850
        self.temp = temp

    # check existing llm
    def check_parameters(self, model_id, prompt_template, refine_template):
        return self.model_id == model_id and self.prompt_template == prompt_template and self.refine_template == refine_template

    # TODO add configuration parameter for temp
    # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=36312850
    # def check_parameters(self, prompt_template, refine_template, temp):
    #     return self.prompt_template == prompt_template & self.refine_template == refine_template & self.temp == temp

    # Replaces acronyms in text with expanded meaning from dictionary
    def replace_acronyms(self, d, text):
        return ' '.join(d[x.upper()] if x.upper() in d else x for x in text.split())

    # Fixes text after preprocessing by adding back punctuation and replacing acronyms
    def fix_text(self, docs):
        acronym_dictionary = get_acronym_dictionary()
        fixed_docs = []
        for text in docs:
            if text.endswith('?'):
                fixed_docs.append(self.replace_acronyms(
                    acronym_dictionary, text.rstrip(punctuation)) + '?')
            elif text.endswith('!'):
                fixed_docs.append(self.replace_acronyms(
                    acronym_dictionary, text.rstrip(punctuation)) + '!')
            elif text.endswith('.') or (text != "" and not text.endswith('.')):
                fixed_docs.append(self.replace_acronyms(
                    acronym_dictionary, text.rstrip(punctuation)) + '.')
            else:
                fixed_docs.append(text)
        return fixed_docs

    # Function to summarize list of texts using LangChain map-reduce chain with custom prompts.
    def get_summary(self, documents):

        summary_text = 'topic summarization disabled'

        if self.llm is None:
            logger.error("TopicSummarizer not initialized")

        elif all(s == '' for s in documents):
            logger.error("no document content to summarize")

        else:
            # replace acronyms and concatenate top n documents
            list_of_texts = '\n'.join(self.fix_text(documents))

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=100)

            # stuffs the lists of text into "Document" objects for LangChain
            docs = text_splitter.create_documents([list_of_texts])

            prompt = PromptTemplate.from_template(self.prompt_template)

            refine_prompt = PromptTemplate.from_template(self.refine_template)

            # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
            chain = load_summarize_chain(self.llm,
                                            chain_type="refine",
                                            verbose=False,
                                            question_prompt=prompt,
                                            refine_prompt=refine_prompt,
                                            return_intermediate_steps=True,
                                            input_key="input_documents",
                                            output_key="output_text",
                                        )

            # saves only the summary (but output includes intermediate steps of how we get to the summary
            # if we want to save that in the future for XAI or other reason e.g., output_summary['intermediate_steps'])
            output_summary = chain({"input_documents": docs})
            summary_text = output_summary['output_text']
            if summary_text == '':
                logger.warning('null output_text in topic summarization')
                summary_text = 'topic summary not available'

        return summary_text


def detect_trending_topics(document_info_train, document_df_test, trend_depth = DEFAULT_TREND_DEPTH_DAYS):

    trending_topic_ids = []

    if not trend_depth:
        logger.info('trending topic detection disabled, depth=0')
        return trending_topic_ids

    elif trend_depth == 1:
        return detect_trending_topics_single_day(document_info_train, document_df_test)

    # filter document_info to last trend_depth days
    trend_docs = document_df_test[document_df_test['Timestamp'] >=
                             document_df_test['Timestamp'].max() - pd.Timedelta(days = trend_depth)]

    topics = set(document_df_test['Topic'])
    for topic_id in topics:

        # count number of posts per day
        topic_docs = trend_docs[trend_docs['Topic'] == topic_id]
        topic_docs['Day'] = topic_docs['Timestamp'].dt.date
        day_counts = topic_docs.groupby(['Day']).agg({'Day': ['count']})
        day_counts.columns = ['num_posts']

        # fill in days with 0 posts
        dates = day_counts.index
        min_date = dates.min()
        max_date = dates.max()
        delta = datetime.timedelta(days=1)
        curr_date = min_date
        while curr_date < max_date:
            d = curr_date
            if d not in dates:
                day_counts = pd.concat([day_counts, pd.DataFrame([{'num_posts': 0}], index=[d])])
            curr_date += delta

        # sort by date
        day_counts = day_counts.sort_index()
        num_posts = day_counts['num_posts']

        # test for trending if the topic has been seen at least twice and within 7 or more days
        if(len(num_posts) > 3):
            stat, p_value = shapiro(num_posts)

            # only include topics that have one day with more than N=5 posts
            if(p_value < DEFAULT_SHAPIRO_P_VALUE and np.max(num_posts) >= DEFAULT_POST_THRESHOLD):
                trending_topic_ids.append(topic_id)

    return trending_topic_ids

def detect_trending_topics_single_day(document_info_train, document_df_test):

    trending_topic_ids = []

    # anomaly detection single day (day with the max posts for given topic)
    topics = set(document_df_test['Topic'])
    for topic_id in topics:

        # count number of posts per day
        test_docs = document_df_test[document_df_test['Topic'] == topic_id]
        test_docs['Day'] = test_docs['Timestamp'].dt.date
        test_day_counts = test_docs.groupby(['Day']).agg({'Day': ['count']})
        test_day_counts.columns = ['num_posts']

        train_docs = document_info_train.loc[document_info_train['Topic'] == topic_id]
        train_docs['Day'] = train_docs['Timestamp'].dt.date
        train_day_counts = train_docs.groupby(['Day']).agg({'Day': ['count']})
        train_day_counts.columns = ['num_posts']

        # get max number of posts for test set
        num_posts = test_day_counts['num_posts']
        test_max = np.max(num_posts)

        # get mean and standard dev of training set
        mean = np.mean(train_day_counts['num_posts'])
        std = np.std(train_day_counts['num_posts'])

        # test if stationary and gaussian
        if(test_max > (mean + DEFAULT_TRAIN_FACTOR * std) and test_max >= DEFAULT_POST_THRESHOLD):
            trending_topic_ids.append(topic_id)

            max_day = test_docs['Day'].values[np.argmax(num_posts)]
            logger.info('peak occurred for topic %d on %s' % (topic_id, max_day))

    return trending_topic_ids

topic_summarizer = TopicSummarizer()
