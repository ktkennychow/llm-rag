import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, DATA_PATH


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {str(semantic_search.model)}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    movies = []
    with open(DATA_PATH, "r") as f:
        movies = json.load(f)
    semantic_search.load_or_create_embeddings(movies["movies"])
    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(
        f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, dict[str, str]] = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def build_embeddings(self, documents: list[dict[str, str]]) -> np.ndarray:
        self.documents = documents
        doc_repr = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_repr.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_repr)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict[str, str]]) -> np.ndarray:
        self.documents = documents
        doc_repr = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_repr.append(f"{doc['title']}: {doc['description']}")
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("Text is empty or contains only whitespace.")
        encoded_text = self.model.encode([text])
        return encoded_text[0]

    def search(self, query: str, limit: int) -> list[dict[str, str]]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        similarities: list[tuple[int, dict[str, str]]] = []
        for doc_id, doc in self.document_map.items():
            similarity = cosine_similarity(query_embedding, self.embeddings[doc_id - 1])
            similarities.append((similarity, doc))
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results: list[dict[str, str]] = []
        for similarity, doc in similarities[:limit]:
            top_results.append(
                {
                    "score": similarity,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
        return top_results
