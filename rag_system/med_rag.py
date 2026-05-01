import json
import time
from gemma_chat import GemmaChat
from openAI_chat import Chat as OpenAIChat
from bioBERT_retriever import BioBERTRetriever
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from medCPT_retriever import MedCPTRetriever

class MedRAG:
    def __init__(
        self,
        retriever=1,
        question_type=1,
        n_docs=10,
        llm_provider="gemma",
        model=None,
        api_key=None,
        retrieval_depth=None,
        chat_client=None,
    ):
        if retriever == 1:
            self.retriever = BioBERTRetriever()
        elif retriever == 2:
            self.retriever = BM25Retriever()
        elif retriever == 3:
            self.retriever = HybridRetriever()
        elif retriever == 4:
            self.retriever = MedCPTRetriever(rerank=True)
        else:
            raise ValueError("Invalid retriever value. Choose 1 for bioBERT, 2 for BM25, 3 for hybrid, or 4 for MedCPT.")

        self.retriever_id = retriever
        self.chat = chat_client or self._build_chat(llm_provider, question_type, model, api_key)
        self.n_docs = n_docs
        self.retrieval_depth = retrieval_depth

    def extract_pmids(self, docs):
        # Extracts PMIDs from the documents and returns them as a list
        return [doc["PMID"] for doc in docs.values()]

    def get_answer(self, question: str) -> str:

        # retrieve the documents timing the retrieval
        start_time_retrieval = time.time()
        retrieved_docs = json.loads(self._retrieve_docs(question))
        end_time_retrieval = time.time()

        # extract the PMIDs from the retrieved documents
        pmids = self.extract_pmids(retrieved_docs)

        # the chat response is a json string {'response': '...', 'used_PMIDs': [...]} and timing the generation
        start_time_generation = time.time()
        answer = self.chat.create_chat(question, retrieved_docs)
        end_time_generation = time.time()

        retrieval_time = end_time_retrieval - start_time_retrieval
        generation_time = end_time_generation - start_time_generation

        # now adding the retrieved PMIDs to the response
        try :
            answer = json.loads(answer)
            answer['retrieved_PMIDs'] = pmids
            answer['retrieval_time'] = retrieval_time
            answer['generation_time'] = generation_time
        except:
            return None
        return json.dumps(answer)

    def _build_chat(self, llm_provider, question_type, model, api_key):
        provider = (llm_provider or "gemma").lower()
        if provider in {"gemma", "google", "google-gemma", "gemini"}:
            return GemmaChat(question_type=question_type, api_key=api_key, model=model)
        if provider in {"openai", "gpt", "gpt-3.5", "gpt-3.5-turbo"}:
            return OpenAIChat(question_type=question_type, api_key=api_key, model=model or "gpt-3.5-turbo")
        raise ValueError("llm_provider must be 'gemma' or 'openai'.")

    def _retrieve_docs(self, question):
        if isinstance(self.retriever, HybridRetriever):
            retrieval_depth = self.retrieval_depth or 50
            return self.retriever.retrieve_docs(question, top_n=self.n_docs, k=retrieval_depth)
        if isinstance(self.retriever, MedCPTRetriever):
            retrieval_depth = self.retrieval_depth or max(self.n_docs, 20)
            return self.retriever.retrieve_docs(question, k=retrieval_depth, top_n=self.n_docs)
        return self.retriever.retrieve_docs(question, self.n_docs)