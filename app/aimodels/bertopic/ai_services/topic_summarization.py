import os
from pathlib import Path
from string import punctuation
from langchain import PromptTemplate
from langchain.llms import CTransformers
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.llamacpp import LlamaCpp
from app.core.logging import logger
from app.core.model_cache import MODEL_CACHE_BASEDIR
from app.core.minio import download_file_from_minio
from app.core.config import get_acronym_dictionary
# from app.aimodels.gpt4all.models.llm_pretrained import LlmPretrainedModel, LlmFilenameEnum
# from app.aimodels.gpt4all.crud import crud_llm_pretrained as crud

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
DEFAULT_LLM_TOP_P = 0.95
DEFAULT_LLM_REPEAT_PENALTY = 1.3
DEFAULT_MAX_NEW_TOKENS = 2000
DEFAULT_CONTEXT_LENGTH = 6000

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
                       temp=DEFAULT_LLM_TEMP,
                       top_p=DEFAULT_LLM_TOP_P,
                       repeat_penalty=DEFAULT_LLM_REPEAT_PENALTY):

        self.model_type = model_obj.model_type
        self.model_id = model_obj.id
        llm_path = os.path.join(MODEL_CACHE_BASEDIR, self.model_type)

        # download gpt4all model binary
        if not os.path.isfile(llm_path):
            # Create the directory if it doesn't exist
            Path(llm_path).parent.mkdir(parents=True, exist_ok=True)

            # Download the file from Minio
            logger.info(f"Downloading model from Minio to {llm_path}")
            download_file_from_minio(model_obj.id, s3, filename=llm_path)

            if not os.path.isfile(llm_path):
                logger.error(f"Error downloading model from Minio to {llm_path}")
            else:
                logger.info(f"Downloaded model from Minio to {llm_path}")

        config = {'max_new_tokens': DEFAULT_MAX_NEW_TOKENS,
                  'temperature': DEFAULT_LLM_TEMP,
                  'context_length': DEFAULT_CONTEXT_LENGTH}
        self.llm = CTransformers(
            model = llm_path,
            model_type='mistral',
            config=config,
            threads=os.cpu_count(),
        )
        self.prompt_template = prompt_template
        self.refine_template = refine_template

        # TODO add configuration parameters for temp, top_p, and repeat_penalty
        # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=36312850
        self.temp = temp
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

    # check existing llm
    def check_parameters(self, model_id, prompt_template, refine_template):
        return self.model_id == model_id and self.prompt_template == prompt_template and self.refine_template == refine_template

    # TODO add configuration parameters for temp, top_p, and repeat_penalty
    # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=36312850
    # def check_parameters(self, prompt_template, refine_template, temp, top_p, repeat_penalty):
    #     return self.prompt_template == prompt_template & self.refine_template == refine_template & self.temp == temp & self.top_p == top_p & self.repeat_penalty == repeat_penalty

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
        if self.llm is None:
            logger.error("TopicSummarizer not initialized")
            return None

        if all(s == '' for s in documents):
            logger.error("no document content to summarize")
            return None

        # replace acronyms and concatenate top n documents
        list_of_texts = '\n'.join(self.fix_text(documents))

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=100)

        docs = text_splitter.create_documents([list_of_texts])  # stuffs the lists of text into "Document" objects for LangChain

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


        return chain({"input_documents": docs})


topic_summarizer = TopicSummarizer()
