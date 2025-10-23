#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

from keyword_search import InvertedIndex, tokenize_str

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# BM25 scoring parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Search result limits
DEFAULT_LIMIT = 5
MAX_RESULTS = 5

# File paths
root_path = Path(__file__).parent.parent

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def load_index_safely(inverted_index: InvertedIndex) -> None:
    """Load the search index with helpful error messages."""
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Error: No search index found. Run 'python cli.py build' first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)


# =============================================================================
# COMMAND HANDLERS
# =============================================================================


def handle_build_command(movies, inverted_index):
    """Build the search index from movies data."""
    inverted_index.build(movies)
    inverted_index.save()
    print("Index built successfully!")


def handle_search_command(args, inverted_index):
    """Handle basic search command."""
    load_index_safely(inverted_index)
    search_results = []
    search_words = tokenize_str(args.query)

    for word in search_words:
        movie_ids = inverted_index.get_documents(word)
        for movie_id in movie_ids:
            movie_obj = inverted_index.docmap.get(movie_id)
            if movie_obj is not None:
                search_results.append(movie_obj)
                if len(search_results) >= MAX_RESULTS:
                    break
        if len(search_results) >= MAX_RESULTS:
            break

    if len(search_results) > 0:
        search_results.sort(key=lambda x: x["id"])

    print(f"Searching for: {args.query}")
    for result in search_results:
        print(f"{result['id']}. {result['title']}")


def handle_bm25_search_command(args, inverted_index):
    """Handle BM25 search command."""
    load_index_safely(inverted_index)
    bm25_results = inverted_index.bm25_search(args.query, args.limit)

    for result_dict in bm25_results:
        for doc_id, score in result_dict.items():
            movie_title = inverted_index.docmap.get(doc_id).get("title")
            print(f"({doc_id}) {movie_title} - Score: {score:.2f}")


def handle_tf_command(args, inverted_index):
    """Handle term frequency command."""
    load_index_safely(inverted_index)
    frequency = inverted_index.get_tf(args.doc_id, args.term)
    print("Frequency: ", frequency)


def handle_bm25_tf_command(args, inverted_index):
    """Handle BM25 term frequency command."""
    load_index_safely(inverted_index)
    bm25_tf_score = inverted_index.get_bm25_tf(args.doc_id, args.term, args.k1)
    print(f"BM25 TF score of '{args.term}': {bm25_tf_score:.2f}")


def handle_idf_command(args, inverted_index):
    """Handle inverse document frequency command."""
    load_index_safely(inverted_index)
    idf_score = inverted_index.get_idf(args.term)
    print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")


def handle_bm25_idf_command(args, inverted_index):
    """Handle BM25 inverse document frequency command."""
    load_index_safely(inverted_index)
    idf_score = inverted_index.get_bm25_idf(args.term)
    print(f"BM25 Inverse document frequency of '{args.term}': {idf_score:.2f}")


def handle_tfidf_command(args, inverted_index):
    """Handle TF-IDF command."""
    load_index_safely(inverted_index)
    tf_score = inverted_index.get_tf(args.doc_id, args.term)
    idf_score = inverted_index.get_idf(args.term)
    tf_idf_score = tf_score * idf_score
    print(
        f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf_score:.2f}"
    )


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main() -> None:
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command - creates the search index
    subparsers.add_parser("build", help="Build the inverted index and document map")

    # Search command - basic search
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Term")

    # Term frequency command - find how often a word appears in a document
    tf_parser = subparsers.add_parser(
        "tf", help="Term Frequency: find frequency of a term in a movie description"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    # BM25 term frequency command - advanced scoring
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get BM25 TF score for a given document ID, term, and optional K1",
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 k1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    # Inverse document frequency command - find how rare a word is
    idf_parser = subparsers.add_parser(
        "idf",
        help="Inverse Document Frequency: find frequency of common words that are specific to our dataset",
    )
    idf_parser.add_argument("term", type=str, help="Term")

    # BM25 inverse document frequency command - advanced rarity scoring
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf",
        help="BM25 Inverse Document Frequency: find frequency of common words that are specific to our dataset",
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term")

    # TF-IDF command - combines term frequency and inverse document frequency
    tf_idf_parser = subparsers.add_parser(
        "tfidf",
        help="Term Frequency-Inverted Document Frequency: find score of a term in a movie description",
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term")

    # BM25 search command - full advanced search
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="Tunable limit"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Load movies data
    with open(root_path / "data" / "movies.json", "r") as f:
        movies_json = json.load(f)
    movies: list[dict[str, str]] = movies_json["movies"]

    # Create search index
    inverted_index = InvertedIndex()

    # Handle different commands using helper functions
    match args.command:
        case "build":
            handle_build_command(movies, inverted_index)
        case "search":
            handle_search_command(args, inverted_index)
        case "bm25search":
            handle_bm25_search_command(args, inverted_index)
        case "tf":
            handle_tf_command(args, inverted_index)
        case "bm25tf":
            handle_bm25_tf_command(args, inverted_index)
        case "idf":
            handle_idf_command(args, inverted_index)
        case "bm25idf":
            handle_bm25_idf_command(args, inverted_index)
        case "tfidf":
            handle_tfidf_command(args, inverted_index)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
