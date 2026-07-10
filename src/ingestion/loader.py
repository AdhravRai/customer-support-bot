import sys
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from src.config.settings import DATA_DIR
from src.utils.logger import logger
from src.utils.exceptions import CustomException


class DocumentLoader:
    """Loads supported documents from the knowledge base."""
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load_documents(self) -> list[Document]:
        """Load all supported documents from the data directory."""
        try:
            documents: list[Document] = []
            for file_path in self.data_dir.iterdir():
                if not file_path.is_file():
                    continue
                suffix = file_path.suffix.lower()
                try:
                    if suffix == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                        loaded_docs=loader.load()
                        documents.extend(loaded_docs)
                        logger.info(f"Loaded PDF: {file_path.name}")   
                    elif suffix == ".txt":
                        loader = TextLoader(str(file_path),encoding="utf-8")
                        loaded_docs=loader.load()
                        documents.extend(loaded_docs)
    
                        logger.info(f"Loaded text file: {file_path.name}")
                    else:
                        logger.warning(f"Skipping unsupported file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}") 
            if not documents:
                raise ValueError(f"No supported documents found inside '{self.data_dir}'.")
        
            logger.info(f"Successfully loaded {len(documents)} document(s).")
            return documents
        except Exception as e:
            raise CustomException(e,sys)