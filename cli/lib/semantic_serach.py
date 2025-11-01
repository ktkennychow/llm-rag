import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, DATA_PATH


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


# def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
#     if len(vec1) != len(vec2):
#         raise ValueError("The lengths of the two vectors are not the same.")
#     sum = []
#     for i in range(len(vec1)):
#         sum.append(vec1[i] + vec2[i])
#     return sum


# def subtract_vectors(vec1, vec2):
#     if len(vec1) != len(vec2):
#         raise ValueError("The lengths of the two vectors are not the same.")
#     sub = []
#     for i in range(len(vec1)):
#         sub.append(vec1[i] - vec2[i])
#     return sub


# def dot(vec1, vec2):
#     if len(vec1) != len(vec2):
#         raise ValueError("vectors must be the same length")
#     total = 0.0
#     for i in range(len(vec1)):
#         total += vec1[i] * vec2[i]
#     return total


# def euclidean_norm(vec):
#     total = 0.0
#     for x in vec:
#         total += x**2

#     return total**0.5


# def cosine_similarity(vec1, vec2):
#     if len(vec1) != len(vec2):
#         raise ValueError("The lengths of the two vectors are not the same.")
#     dots = []
#     for i in range(len(vec1)):
#         dots.append(vec1[i] * vec2[i])
#     total = 0
#     for d in dots:
#         total += d

#     mag1 = euclidean_norm(vec1)
#     mag2 = euclidean_norm(vec2)
#     if mag1 == 0 or mag2 == 0:
#         return 0.0
#     return total / (mag1 * mag2)
