import sys
from langchain_community.vectorstores import FAISS
from src.config.settings import (
    VECTORSTORE_DIR,
    SEARCH_TYPE,
    TOP_K,
)
from src.embeddings.embedding_model import EmbeddingModel
from src.utils.logger import logger
from src.utils.exceptions import CustomException

class VectorStoreRetriever:
    """Loads the FAISS vector store and returns a retriever."""
    def __init__(self):
        self.embedding_model = EmbeddingModel()
    def get_retriever(self):
        try:
            embeddings = self.embedding_model.get_embeddings()
            vector_store = FAISS.load_local(
                str(VECTORSTORE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            retriever = vector_store.as_retriever(
                search_type=SEARCH_TYPE,
                search_kwargs={"k": TOP_K},
            )
            logger.info("Retriever loaded successfully.")
            return retriever
        except Exception as e:
            raise CustomException(e, sys)