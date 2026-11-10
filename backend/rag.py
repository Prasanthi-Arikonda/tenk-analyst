"""Shared building blocks: the embedding model, the Qdrant connection, and filters.

Ingest (stage 0) and all later pipelines import from here, so they all use
the same models and the same collection.
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient, models

load_dotenv()  # reads the keys from the .env file into environment variables

OPENROUTER = "https://openrouter.ai/api/v1"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tenk")

# An "embedding" turns text into a list of 1536 numbers. Similar texts get similar numbers.
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url=OPENROUTER,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    check_embedding_ctx_length=False,  # required when calling OpenRouter
)

# The chat model that reads the question and writes answers. temperature=0 keeps answers steady.
llm = ChatOpenAI(
    model=os.environ.get("CHAT_MODEL", "anthropic/claude-haiku-5.5"),
    base_url=OPENROUTER,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0,
    max_tokens=1500,
)

client = QdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"))


def ensure_collection() -> None:
    """Create the collection (and the indexes we filter on) if it does not exist yet."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    # LangChain stores metadata under "metadata.", so the keys start with that.
    client.create_payload_index(COLLECTION, "metadata.company", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.item", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.year", models.PayloadSchemaType.INTEGER)


def get_store() -> QdrantVectorStore:
    """The LangChain wrapper around our Qdrant collection."""
    return QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)


@traceable(run_type="retriever", name="qdrant_search")  # shows up as its own step in LangSmith
def search(question: str, k: int = 5, flt=None) -> list:
    """Vector search in Qdrant: the k chunks closest to the question, optionally filtered."""
    return get_store().similarity_search(question, k=k, filter=flt)


def make_filter(companies=None, years=None, items=None):
    """Build a Qdrant filter. An empty or missing list means 'no limit' on that field."""
    must = []
    if companies:
        must.append(models.FieldCondition(key="metadata.company", match=models.MatchAny(any=companies)))
    if years:
        must.append(models.FieldCondition(key="metadata.year", match=models.MatchAny(any=years)))
    if items:
        must.append(models.FieldCondition(key="metadata.item", match=models.MatchAny(any=items)))
    return models.Filter(must=must) if must else None


def count_points(companies=None, years=None) -> int:
    """How many chunks are stored (optionally for some companies and years)."""
    result = client.count(COLLECTION, count_filter=make_filter(companies, years), exact=True)
    return result.count


# ---- Reranking ----
# FlashRank is a small model (about 3 MB) that runs on the CPU. It is loaded once, on first use.
_ranker = None


@traceable(name="flashrank_rerank")
def rerank(question: str, docs: list, top_n: int = 5) -> list:
    """Score each chunk against the question and keep the best top_n, best first."""
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    from flashrank import RerankRequest
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    ranked = _ranker.rerank(RerankRequest(query=question, passages=passages))
    return [docs[r["id"]] for r in ranked[:top_n]]
