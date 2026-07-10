import sys
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import GraphState
from src.vector_store.retriever import VectorStoreRetriever
from src.prompts.prompt import RAG_PROMPT
from src.config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TEMPERATURE,
)

from src.utils.logger import logger
from src.utils.exceptions import CustomException
from pathlib import Path

class GraphNodes:
    def __init__(self):

        self.retriever = VectorStoreRetriever()
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=TEMPERATURE,
        )
        self.parser = StrOutputParser()
    def retrieve(self, state: GraphState):
        try:
            retriever = self.retriever.get_retriever()    
            documents = retriever.invoke(state["question"])    
            logger.info(f"Retrieved {len(documents)} documents.")
    
            return {
                "context": documents
            }    
        except Exception as e:
            raise CustomException(e, sys)


    def generate(self, state: GraphState):
        try:
            context = "\n\n".join(
                f"Source: {Path(doc.metadata.get('source', 'Unknown')).name}\n"
                f"{doc.page_content}"
                for doc in state["context"]
            )    
            sources = []
            for doc in state["context"]:
                source = doc.metadata.get("source")           
                if source:
                    source_name = Path(source).name            
                    if source_name not in sources:
                        sources.append(source_name)
            if not context.strip():
                return {
                    "answer": (
                        "I couldn't find relevant information in the knowledge base."
                    ),
                    "sources": [],
                }

            history = []            
            for message in state["chat_history"]:
                history.append(
                    f"{message.type}: {message.content}"
                )
            chat_history = "\n".join(history)
            chain = (RAG_PROMPT | self.llm | self.parser)   
            response = chain.invoke(
                {
                    "context": context,
                    "chat_history": chat_history,
                    "question": state["question"],
                }
            )    
            logger.info("Generated response.")
            return {
                "answer": response,
                "sources": sources,
                "context": state["context"],
            }
    
        except Exception as e:
            raise CustomException(e, sys)