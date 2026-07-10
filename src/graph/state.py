from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    question: str
    context: list[Document]
    answer: str
    sources: list[str]
    chat_history: list[BaseMessage]