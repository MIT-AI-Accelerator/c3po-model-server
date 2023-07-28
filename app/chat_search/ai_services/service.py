import re
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from app.aimodels.gpt4all.ai_services.completion_inference import (
    CompletionInference,
    CompletionInferenceInputs,
)
from app.chat_search.retriever import MarcoRerankRetriever
from app.core.errors import ValidationError
from app.core.model_cache import MODEL_CACHE_BASEDIR

from sample_data import CHAT_DATASET_1_PATH

class RetrievalService:
    def __init__(self, completion_inference: CompletionInference):
        # validate input
        self.completion_inference = completion_inference

    def retrieve(self, api_inputs: CompletionInferenceInputs, summarize=False):
        # validate input
        if not isinstance(api_inputs, CompletionInferenceInputs):
            raise ValidationError("must input type CompletionInferenceInputs")

        retriever = self._build_retriever(channel_names=[CHAT_DATASET_1_PATH])

        if summarize:
            llm = self.completion_inference._build_llm(api_inputs)
            results = self._retrieve_and_summarize(
                llm,
                query=api_inputs.prompt,
                retriever=retriever,
            )
        else:
            results = self._retrieve_only(
                query=api_inputs.prompt,
                retriever=retriever,
            )

        return results

    def _build_retriever(
        self,
        channel_names=[],
    ):
        # TODO: pull from minio
        # model_name = os.path.join(MODEL_CACHE_BASEDIR, "all-MiniLM-L6-v2")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        local_embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # name = channel_names[0]
        path = channel_names[0]
        # path = os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
        chat_texts = CSVLoader(path).load()
        chat_retriever = FAISS.from_documents(
            chat_texts, local_embeddings
        ).as_retriever()
        chat_retriever.search_kwargs = {"k": 25}

        def bm25_preprocess_func(text):
            # replace non alphanumeric characters with whitespace
            text = re.sub(r"[^a-zA-Z0-9]", " ", text)

            # lowercase and split on whitespace
            return text.lower().split()

        # initialize the bm25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            chat_texts, preprocess_func=bm25_preprocess_func
        )
        bm25_retriever.k = 25

        # initialize the ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chat_retriever], weights=[0.5, 0.5]
        )

        rerank_retriever = MarcoRerankRetriever(
            base_retriever=ensemble_retriever,
            rerank_model_name_or_path="cross-encoder/ms-marco-TinyBERT-L-6",
            max_relevant_documents=10,
        )

        return rerank_retriever

    def _retrieve_only(self, query=None, retriever=None):
        result = {"input": query, "result": "No LLM used to summarize"}
        result["source_documents"] = retriever.get_relevant_documents(query)
        return result

    def _retrieve_and_summarize(self, llm, query=None, retriever=None):
        ###Unknown: how to address FAISS chunking and add metadata
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="input",
            return_source_documents=True,
            verbose=True,
        )

        result = chain({"input": f"{query}"})
        return result
