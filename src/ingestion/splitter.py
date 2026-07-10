import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.utils.logger import logger
from src.utils.exceptions import CustomException

class DocumentSplitter:
    """Splits documents into smaller chunks for retrieval."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split_documents(self,documents:list[Document])->list[Document]:
        try:
            text_splitter=RecursiveCharacterTextSplitter(chunk_overlap=self.chunk_overlap,chunk_size=self.chunk_size)
            chunks=text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            raise CustomException(e,sys)