from typing import Union
from pydantic import BaseModel, UUID4
from fastapi import Depends, APIRouter
from minio import Minio
from app.dependencies import get_db, get_minio
from sqlalchemy.orm import Session
from .. import crud
from app.core.errors import ValidationError, HTTPValidationError
from ..models import Gpt4AllPretrainedModel
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

router = APIRouter(
    prefix="/completions"
)

class InputRequest(BaseModel):
    gpt4all_pretrained_id: UUID4
    prompt: str

class OutputResponse(BaseModel):
    completion: str

@router.post(
    "/",
    response_model=Union[OutputResponse, HTTPValidationError],
    responses={'422': {'model': HTTPValidationError}},
    summary="GPT completion endpoint",
    response_description="Completed GPT response"
)
def gpt_completion_post(request: InputRequest, db: Session = Depends(get_db), s3: Minio = Depends(get_minio)) -> (
    Union[OutputResponse, HTTPValidationError]
):
    """
    GPT completion endpoint.
    """
    # check to make sure id exists
    gpt4all_pretrained_obj: Gpt4AllPretrainedModel = crud.gpt4all_pretrained.get(
        db, request.gpt4all_pretrained_id)
    if not gpt4all_pretrained_obj:
        return HTTPValidationError(detail=[ValidationError(loc=['path', 'gpt4all_pretrained model upload'], msg='Invalid pretrained model id', type='value_error')])

    # check to make sure gpt4all_pretrained_obj has and embedding layer
    if not gpt4all_pretrained_obj.uploaded:
        return HTTPValidationError(detail=[ValidationError(loc=['path', 'gpt4all_pretrained upload'], msg='gpt4all_pretrained model type has not been uploaded', type='value_error')])


    # perform inference
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])



    local_path = './models/ggml-gpt4all-l13b-snoozy.bin'  # replace with your desired local file path
    
    import requests

    from pathlib import Path
    from tqdm import tqdm

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Example model. Check https://github.com/nomic-ai/gpt4all for the latest models.
    url = 'http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin'

    # send a GET request to the URL to download the file. Stream since it's large
    response = requests.get(url, stream=True)

    # open the file in binary mode and write the contents of the response to it in chunks
    # This is a large file, so be prepared to wait.
    with open(local_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
    llm = GPT4All(model=local_path, backend='gptj', callbacks=callbacks, verbose=True)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

    llm_chain.run(question)


    # basic_inference = BasicInference(gpt4all_pretrained_obj, s3)
    # inference_output = basic_inference.train_bertopic_on_documents(
    #     documents, precalculated_embeddings=precalculated_embeddings, num_topics=1)
    # new_plotly_bubble_config = inference_output.plotly_bubble_config

    


    return new_bertopic_trained_obj
