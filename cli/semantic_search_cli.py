#!/usr/bin/env python3

import argparse
import json
import re

from lib.search_utils import DATA_PATH
from lib.semantic_serach import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify model")
    subparsers.add_parser("verify_embeddings", help="verify embeddings")

    embed_text_parser = subparsers.add_parser("embed_text", help="embed text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_text_parser = subparsers.add_parser(
        "embedquery", help="embed query text"
    )
    embed_query_text_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="semantic search")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Limit the number of results"
    )

    chuck_parser = subparsers.add_parser("chunk", help="chunk text")
    chuck_parser.add_argument("text", type=str, help="Text to chunk")
    chuck_parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size")
    chuck_parser.add_argument("--overlap", type=int, default=0, help="Overlap size")

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="semantic chunk text"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Max chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap size"
    )

    args = parser.parse_args()
    match args.command:
        case "chunk":
            words = args.text.split()
            chunks = []
            cur_idx = 0
            while cur_idx < len(words):
                end_idx = cur_idx + args.chunk_size
                if end_idx > len(words):
                    end_idx = len(words)
                chunk = words[cur_idx:end_idx]
                chunks.append(" ".join(chunk))
                if end_idx >= len(words):
                    break
                cur_idx = end_idx - args.overlap
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        case "semantic_chunk":
            sentences = re.split(r"(?<=[.!?])\s+", args.text)
            chunks = []
            cur_idx = 0
            while cur_idx < len(sentences):
                end_idx = cur_idx + args.max_chunk_size
                if end_idx > len(sentences):
                    end_idx = len(sentences)
                chunk = sentences[cur_idx:end_idx]
                chunks.append(" ".join(chunk))
                if end_idx >= len(sentences):
                    break
                cur_idx = end_idx - args.overlap
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
            return chunks
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            movies = []
            with open(DATA_PATH, "r") as f:
                movies = json.load(f)
            semantic_search.load_or_create_embeddings(movies["movies"])
            results = semantic_search.search(args.query, args.limit)

            for i, result in enumerate(results):
                print(f"{i + 1}. {result['title']} (score: {result['score']})")
                print(f"    {result['description']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
