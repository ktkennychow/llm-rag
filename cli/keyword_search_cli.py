#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

from keyword_search import InvertedIndex, tokenize_str


root_path = Path(__file__).parent.parent
BM25_K1 = 1.5
BM25_B = 0.75


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index and document map")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Term")

    tf_parser = subparsers.add_parser(
        "tf", help="Term Frequency: find frequency of a term in a movie description"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

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

    idf_parser = subparsers.add_parser(
        "idf",
        help="Inverse Document Frequency: find frequency of common words that are specific to our dataset",
    )
    idf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf",
        help="BM25 Inverse Document Frequency: find frequency of common words that are specific to our dataset",
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term")

    tf_idf_parser = subparsers.add_parser(
        "tfidf",
        help="Term Frequency-Inverted Document Frequency: find score of a term in a movie description",
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term")

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Tunable limit")

    args = parser.parse_args()

    with open(root_path / "data" / "movies.json", "r") as f:
        movies_json = json.load(f)
    movies: list[dict[str, str]] = movies_json["movies"]

    inverted_index = InvertedIndex(
        index={}, docmap={}, term_frequencies={}, doc_lengths={}
    )

    match args.command:
        case "build":
            inverted_index.build(movies)
            inverted_index.save()
        case "search":
            try:
                inverted_index.load()
            except Exception as e:
                print("Error:", e)
                sys.exit("exit")
            list_of_matches = []
            query_tokens = tokenize_str(args.query)
            for qt in query_tokens:
                movie_ids = inverted_index.get_documents(qt)
                for id in movie_ids:
                    movie_obj = inverted_index.docmap.get(id)
                    if movie_obj is not None:
                        list_of_matches.append(movie_obj)
                        if len(list_of_matches) >= 5:
                            break
                if len(list_of_matches) >= 5:
                    break
            if len(list_of_matches) > 0:
                list_of_matches.sort(key=lambda x: x["id"])

            print(f"Searching for: {args.query}")
            for match in list_of_matches:
                print(f"{match["id"]}. {match["title"]}")
        case "bm25search":
            try:
                inverted_index.load()
            except Exception as e:
                print("Error:", e)
                sys.exit("exit")
            list_bm25 = inverted_index.bm25_search(args.query, args.limit)
            for dict in list_bm25:
                for doc_id, score in dict.items():
                    name = inverted_index.docmap.get(doc_id).get("title")
                    print(f"({doc_id}) {name} - Score: {score:.2f}")

        case "tf":
            try:
                inverted_index.load()
            except Exception as e:
                print("Error:", e)
                sys.exit("exit")
            tf = inverted_index.get_tf(args.doc_id, args.term)
            print("Frequency: ", tf)
        case "bm25tf":
            try:
                inverted_index.load()
            except Exception as e:
                print("Error:", e)
                sys.exit("exit")
            bm25tf = inverted_index.get_bm25_tf(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}': {bm25tf:.2f}")
        case "idf":
            try:
                inverted_index.load()
            except Exception as e:
                print("Error:", e)
                sys.exit("exit")
            idf = inverted_index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "bm25idf":
            try:
                inverted_index.load()
            except Exception as e:
                print("Error:", e)
                sys.exit("exit")
            idf = inverted_index.get_bm25_idf(args.term)
            print(f"BM25 Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try:
                inverted_index.load()
            except Exception as e:
                print("Error:", e)
                sys.exit("exit")
            tf = inverted_index.get_tf(args.doc_id, args.term)
            idf = inverted_index.get_idf(args.term)
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
