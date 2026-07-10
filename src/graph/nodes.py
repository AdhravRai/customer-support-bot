import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import GraphState
from src.vector_store.retriever import VectorStoreRetriever
from src.prompts.prompt import RAG_PROMPT
from src.config.settings import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    TEMPERATURE,
)

from src.utils.logger import logger
from src.utils.exceptions import CustomException

class GraphNodes:
    def __init__(self):

        self.retriever = VectorStoreRetriever()
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
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
                doc.page_content
                for doc in state["context"]
            )    
            chain = (RAG_PROMPT | self.llm | self.parser)   
            response = chain.invoke(
                {
                    "context": context,
                    "question": state["question"],
                }
            )    
            logger.info("Generated response.")
            return {
                "answer": response
            }
    
        except Exception as e:
            raise CustomException(e, sys)