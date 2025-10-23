#!/usr/bin/env python3

from collections import Counter
import math
import os
from pathlib import Path
import pickle
import string
from nltk.stem import PorterStemmer

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# BM25 scoring parameters
BM25_K1 = 1.5
BM25_B = 0.75

# File paths
root_path = Path(__file__).parent.parent


# =============================================================================
# TEXT PROCESSING FUNCTIONS
# =============================================================================


def tokenize_str(txt: str) -> list[str]:
    """Convert text to searchable tokens by cleaning and processing."""
    # Step 1: Remove punctuation and make lowercase
    punctuation = string.punctuation
    translation_table = str.maketrans("", "", punctuation)
    clean_text = txt.lower().translate(translation_table)

    # Step 2: Split into words
    words = clean_text.split()

    # Step 3: Remove empty words
    words = remove_empty_tokens(words)

    # Step 4: Remove stop words (common words like 'the', 'and', etc.)
    words = remove_stop_words(words)

    # Step 5: Apply stemming (reduce words to root form)
    words = apply_stemming(words)

    return words


def remove_empty_tokens(tokens: list[str]) -> list[str]:
    """Remove empty strings from token list."""
    return [token for token in tokens if token != ""]


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Remove common words that don't add meaning to search."""
    with open(root_path / "data" / "stopwords.txt", "r") as f:
        stopwords_text = f.read()

    stopwords = stopwords_text.splitlines()
    return [token for token in tokens if token not in stopwords]


def apply_stemming(tokens: list[str]) -> list[str]:
    """Reduce words to their root form (e.g., 'running' -> 'run')."""
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


# =============================================================================
# SEARCH INDEX CLASS
# =============================================================================


class InvertedIndex:
    """Main search index that stores documents and enables fast searching."""

    def __init__(self):
        # Maps words to document IDs that contain them
        self.index: dict[str, set[int]] = {}
        # Maps document IDs to movie information
        self.docmap: dict[int, dict[str, str]] = {}
        # Counts how many times each word appears in each document
        self.term_frequencies: dict[int, Counter] = {}
        # Stores the length (word count) of each document
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str):
        """Add a document to the search index."""
        text_tokens = tokenize_str(text)

        # Build inverted index - map each word to documents that contain it
        for token in set(text_tokens):
            self.index.setdefault(token, set()).add(doc_id)

        # Count how many times each word appears in this document
        self.term_frequencies[doc_id] = Counter(text_tokens)

        # Store document length for BM25 scoring
        self.doc_lengths[doc_id] = len(text_tokens)

    def __get_avg_doc_length(self) -> float:
        """Calculate average document length for BM25 scoring."""
        num_documents = len(self.doc_lengths)
        if num_documents == 0:
            return 0.0

        total_length = sum(self.doc_lengths.values())
        return total_length / num_documents

    def get_documents(self, term: str) -> list[int]:
        """Get all document IDs that contain the given term."""
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        """Get how many times a word appears in a specific document."""
        search_tokens = tokenize_str(term)
        if len(search_tokens) != 1:
            raise ValueError("Term includes more than one token.")
        term_counter = self.term_frequencies.get(doc_id, {})
        return term_counter.get(search_tokens[0], 0)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1) -> float:
        """Get BM25 term frequency score for a word in a document."""
        raw_tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_normalization = 1 - BM25_B + BM25_B * (doc_length / avg_doc_length)
        bm25_tf_score = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_normalization)
        return bm25_tf_score

    def get_idf(self, term: str) -> float:
        """Get inverse document frequency - how rare a word is across all documents."""
        search_tokens = tokenize_str(term)
        if len(search_tokens) != 1:
            raise ValueError("Term includes more than one token.")
        total_docs = len(self.docmap)
        docs_with_term = len(self.index.get(search_tokens[0], set()))
        return math.log((total_docs + 1) / (docs_with_term + 1))

    def get_bm25_idf(self, term: str) -> float:
        """Get BM25 inverse document frequency score."""
        search_tokens = tokenize_str(term)
        if len(search_tokens) != 1:
            raise ValueError("Term includes more than one token.")
        total_docs = len(self.docmap)
        docs_with_term = len(self.index.get(search_tokens[0], set()))
        return math.log(
            (total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1
        )

    def bm25(self, doc_id: int, term: str) -> float:
        """Calculate full BM25 score for a term in a document."""
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int):
        """Search using BM25 scoring and return top results."""
        query_tokens = tokenize_str(query)
        document_scores: dict[int, float] = {}

        for search_word in query_tokens:
            if search_word not in self.index:
                continue
            for doc_id in self.index[search_word]:
                document_scores.setdefault(doc_id, 0)
                bm25_score = self.bm25(doc_id, search_word)
                document_scores[doc_id] += bm25_score

        # Sort by score and return top results
        sorted_results = [
            {doc_id: score}
            for doc_id, score in sorted(
                document_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return sorted_results[:limit]

    # =============================================================================
    # INDEX BUILDING AND CACHING FUNCTIONS
    # =============================================================================

    def build(self, movies: list[dict[str, str]]):
        """Build the search index from a list of movies."""
        for movie in movies:
            # Store movie information
            self.docmap[movie["id"]] = movie
            # Add movie text to search index
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)

    def save(self):
        """Save the search index to cache files."""
        os.makedirs(root_path / "cache", exist_ok=True)

        # Save each component of the index
        with open(root_path / "cache" / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(root_path / "cache" / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(root_path / "cache" / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(root_path / "cache" / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        """Load the search index from cache files."""
        # Load word-to-documents mapping
        with open(root_path / "cache" / "index.pkl", "rb") as f:
            stored_index = pickle.load(f)
            if stored_index is None:
                raise FileNotFoundError("There is no stored index.")
            self.index = stored_index

        # Load document information
        with open(root_path / "cache" / "docmap.pkl", "rb") as f:
            stored_docmap = pickle.load(f)
            if stored_docmap is None:
                raise FileNotFoundError("There is no stored docmap.")
            self.docmap = stored_docmap

        # Load term frequency counts
        with open(root_path / "cache" / "term_frequencies.pkl", "rb") as f:
            stored_term_frequencies = pickle.load(f)
            if stored_term_frequencies is None:
                raise FileNotFoundError("There is no stored term frequencies.")
            self.term_frequencies = stored_term_frequencies

        # Load document lengths
        with open(root_path / "cache" / "doc_lengths.pkl", "rb") as f:
            stored_doc_lengths = pickle.load(f)
            if stored_doc_lengths is None:
                raise FileNotFoundError("There is no stored doc lengths.")
            self.doc_lengths = stored_doc_lengths
