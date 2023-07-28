# follow pattern here:  https://python.langchain.com/docs/modules/chains/additional/multi_retrieval_qa_router
import os
from langchain import ConversationChain, LLMChain
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.prompt import DEFAULT_TEMPLATE
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from app.aimodels.gpt4all.ai_services.completion_inference import (
    CompletionInference,
    CompletionInferenceInputs,
)
from app.core.errors import ValidationError


class RetrievalService:
    def __init__(self, completion_inference: CompletionInference):
        # validate input
        self.completion_inference = completion_inference

    def question_response(self, api_inputs: CompletionInferenceInputs):
        return self._general_retrieval_base(api_inputs)

    def _general_retrieval_base(
        self,
        api_inputs: CompletionInferenceInputs,
        template: str = """{api_prompt}""",
    ):
        # validate input
        if not isinstance(api_inputs, CompletionInferenceInputs):
            raise ValidationError("must input type CompletionInferenceInputs")

        # build the prompt template
        prompt = PromptTemplate(template=template, input_variables=["api_prompt"])

        # build the chain
        llm = self.completion_inference._build_llm(api_inputs)
        self._perform_retrieval(
            llm, query=api_inputs.prompt, channel_names=["(CUI) MM_chats_20230620.csv"]
        )

    def _perform_retrieval(
        self,
        llm,
        query=None,
        embedding=None,
        channel_names=["(CUI) MM_chats_20230620.csv"],
    ):
        # cache_folder
        #         self.client = sentence_transformers.SentenceTransformer(
        #             self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        #         )

        # TODO: pull from minio
        local_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        personal_texts = [
            "I love apple pie",
            "My favorite color is fuchsia",
            "My dream is to become a professional dancer",
            "I broke my arm when I was 12",
            "My parents are from Peru",
        ]
        personal_retriever = FAISS.from_texts(
            personal_texts, local_embeddings
        ).as_retriever()

        retriever_infos = [
            # {
            #     "name": "personal",
            #     "description": "Good for answering questions about me",
            #     "retriever": personal_retriever,
            # },
        ]

        for name in channel_names:
            path = os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
            chat_docs = CSVLoader(path).load()
            # chat_docs = TextLoader(path).load()
            chat_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chat_texts = chat_text_splitter.split_documents(chat_docs)
            chat_retriever = FAISS.from_documents(
                chat_texts, local_embeddings
            ).as_retriever()

            retriever_info = {
                "name": "AMC",
                "description": f"Good for answering questions about anything having to do with {name}",
                "retriever": chat_retriever,
            }

            retriever_infos.append(retriever_info)

            # Step 1: Define a retriever that inputs a vectorstore-backed retreiver, and can set / update the vector store retriever
            # Step 2: add in a method in new retreiver that reranks based on marco or input reranker of sentence transformers
            # Step 3: return in the get_documents methods with rerank

            



            # Step 4: add in BM25 retriever
            # Step 5: build just diffcse, just epubs, just bm25, ensemble of 2, ensemble of 3 for results
            # step 6: pass to LLM
            ###Unknown: how to address FAISS chunking and add metadata

        # prompt_template = DEFAULT_TEMPLATE.replace("input", "query")
        # prompt = PromptTemplate(
        #     template=prompt_template, input_variables=["history", "query"]
        # )
        # default_chain = LLMChain(
        #     llm=llm,
        #     prompt=prompt,
        # )

        # default_chain.output_key="result"

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_infos[0]["retriever"],
            input_key="input",
            return_source_documents=True,
            verbose=True,
        )

        # chain = MultiRetrievalQAChain.from_retrievers(
        #     llm,
        #     retriever_infos,
        #     verbose=True,
        #     default_chain=default_chain,
        #     # handle_parsing_errors=_handle_error,
        # )

        # for key in chain.destination_chains:
        #     temp = chain.destination_chains[key]
        #     temp.return_source_documents = True
        #     chain.destination_chains[key] = temp

        result = chain({"input": "Summarize all your information on RCH715."})

        with open("answers.txt", "w") as f:
            f.write(f"\ninput: {result['input']}\n")
            f.write(f"result: {result['result']}")
            f.write("\n\n***********SOURCES******************\n")

            for index, doc in enumerate(result["source_documents"]):
                f.write(f"\n\n(source #{index}): {doc.page_content}")
