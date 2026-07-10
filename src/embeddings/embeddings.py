from langchain_huggingface import HuggingFaceEmbeddings
from src.config.settings import EMBEDDING_MODEL


class EmbeddingModel:
    """Creates the embedding model used by the RAG pipeline."""

    def __init__(self):
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logger.info(f"Initialized embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            raise CustomException(e,sys)        
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Returns the initialized embedding model.
        """
        return self.embedding_model
