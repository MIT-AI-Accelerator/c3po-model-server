import os
from pydantic import BaseModel, validator
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pathlib import Path
from app.core.minio import download_file_from_minio
from minio import Minio
from app.core.model_cache import MODEL_CACHE_BASEDIR
from app.core.logging import logger, LogConfig
from logging.config import dictConfig
from ..models import Gpt4AllPretrainedModel

dictConfig(LogConfig().dict())

BASE_CKPT_DIR = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "./data")

class InitInputs(BaseModel):
    gpt4all_pretrained_model_obj: Gpt4AllPretrainedModel
    s3: Minio

    # ensure that model type is defined
    @validator('gpt4all_pretrained_model_obj')
    def gpt4all_pretrained_model_obj_must_have_model_type_and_be_uploaded(cls, v):
        # pylint: disable=no-self-argument
        if not v.model_type:
            raise ValueError(
                'gpt4all_pretrained_model_obj must have model_type')
        if not v.uploaded:
            raise ValueError(
                'gpt4all_pretrained_model_obj must be uploaded')

        return v

    class Config:
        arbitrary_types_allowed = True


class CompletionInferenceOutputs(BaseModel):
    completion: str = ""

    class Config:
        arbitrary_types_allowed = True


class BasicResponseInputs(BaseModel):
    input: str

    class Config:
        arbitrary_types_allowed = True


class CompletionInference:

    def __init__(self, gpt4all_pretrained_model_obj, s3):

        # validate input
        InitInputs(
            gpt4all_pretrained_model_obj=gpt4all_pretrained_model_obj, s3=s3
        )

        self.llm_path = os.path.join(MODEL_CACHE_BASEDIR, gpt4all_pretrained_model_obj.model_type)

        if not os.path.isfile(self.llm_path):
            # Create the directory if it doesn't exist
            Path(self.llm_path).parent.mkdir(parents=True, exist_ok=True)

            # Download the file from Minio
            logger.info(f"Downloading model from Minio to {self.llm_path}")
            download_file_from_minio(gpt4all_pretrained_model_obj.id, s3, filename=self.llm_path)
            logger.info(f"Downloaded model from Minio to {self.llm_path}")

    def basic_response(self, input):

        # validate input
        BasicResponseInputs(input=input)

        # build the template
        template = """Input: {input}

        Response: """
        prompt = PromptTemplate(template=template, input_variables=["input"])

        # Callbacks support token-wise streaming
        callbacks = [StreamingStdOutCallbackHandler()]

        # Verbose is required to pass to the callback manager
        # TODO If you want to use a custom model add the backend parameter (e.g., backend='gptj'),
        # see https://docs.gpt4all.io/gpt4all_python.html
        llm = GPT4All(model=self.llm_path, callbacks=callbacks, verbose=False)

        # run inference
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        completion = llm_chain.run(input)

        return CompletionInferenceOutputs(completion=completion)

    def question_response(self, input):

        # validate input
        BasicResponseInputs(input=input)

        # build the template
        template = """Question: {input}

        Answer: Let's think step by step."""
        prompt = PromptTemplate(template=template, input_variables=["input"])

        # Callbacks support token-wise streaming
        callbacks = [StreamingStdOutCallbackHandler()]

        # Verbose is required to pass to the callback manager
        # TODO If you want to use a custom model add the backend parameter (e.g., backend='gptj'),
        # see https://docs.gpt4all.io/gpt4all_python.html
        llm = GPT4All(model=self.llm_path, callbacks=callbacks, verbose=False)

        # run inference
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        completion = llm_chain.run(input)

        return CompletionInferenceOutputs(completion=completion)
