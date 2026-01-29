from pydantic import BaseModel, Field

from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import OfflineRAG

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the agent")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the agent")

def build_rag_chain(llm, data_dir, data_type):
    doc_loader = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(documents=doc_loader).get_retriever()
    rag_chain = OfflineRAG(llm).get_chain(retriever)

    return rag_chain

