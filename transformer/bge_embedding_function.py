from chromadb import EmbeddingFunction, Documents, Embeddings
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Optional, Dict, List
import os
import uvicorn
from fastapi import FastAPI

app = FastAPI()

DEFAULT_BGE_MODEL = "BAAI/bge-base-en"
DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)


class BgeEmbeddingFunction(BaseModel):
    """HuggingFace BGE sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceBgeEmbeddings

            model_name = "BAAI/bge-large-en"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Optional[Any] = None
    model_name: str = DEFAULT_BGE_MODEL
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    """Instruction to use for embedding query."""
    
    model_config = ConfigDict(extra='ignore')
    model_config['protected_namespaces'] = ()

    def __init__(self, **kwargs: Optional[Any]):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import sentence_transformers

        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence_transformers`."
            ) from exc

        self.client = sentence_transformers.SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    def embed(self, texts: Documents) -> Embeddings:
    #the func accepts an input of type Document and returns a data of type Embedding
        """Compute doc embeddings using a HuggingFace transformer model.

            Args:
                texts: The list of texts to embed.

            Returns:
                List of embeddings, one for each text.
            """
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.client.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()


    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client.encode(
            self.query_instruction + text, **self.encode_kwargs
        )
        return embedding.tolist()



bge_emb = BgeEmbeddingFunction()

@app.get("/")
def welcome():
    return "Try the embedding func"


@app.post("/v1/embed")
def new(texts: Documents) -> Embeddings:
#the func accepts an input of type Document and returns a data of type Embedding
    """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
    embeddings = bge_emb.embed(texts)
    return embeddings


@app.get("/health")
def healthcheck():
    return "ok"