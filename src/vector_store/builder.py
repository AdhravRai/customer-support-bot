import sys
from langchain_community.vectorstores import FAISS
from src.config.settings import VECTORSTORE_DIR
from src.ingestion.loader import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.embeddings.embedding_model import EmbeddingModel

from src.utils.logger import logger
from src.utils.exceptions import CustomException
class VectorStoreBuilder:
    """Builds and saves the FAISS vector store."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()
        self.embedding_model = EmbeddingModel()

    def build_vector_store(self) -> FAISS:
        try:
            documents = self.loader.load_documents()
            chunks = self.splitter.split_documents(documents)
            embeddings = self.embedding_model.get_embeddings()
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            VECTORSTORE_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            vector_store.save_local(str(VECTORSTORE_DIR))

            logger.info("Vector store created successfully.")
            return vector_store
        except Exception as e:
            raise CustomException(e, sys)