#!/usr/bin/env python3

import math
import os
from pathlib import Path
import pickle
import string
from nltk.stem import PorterStemmer

root_path = Path(__file__).parent.parent
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize_str(txt: str) -> list[str]:
    punc = string.punctuation
    trans_table = str.maketrans("", "", punc)
    trans_splited_txt: str = txt.lower().translate(trans_table).split()
    tokens = stemming(rm_stop_words(rm_empty_tokens(trans_splited_txt)))
    return tokens


def rm_empty_tokens(tokens: list[str]) -> list[str]:
    res = []
    for tk in tokens:
        if tk != "":
            res.append(tk)
    return res


def rm_stop_words(tokens: list[str]) -> list[str]:
    with open(root_path / "data" / "stopwords.txt", "r") as f:
        stopwords_text = f.read()

    stopwords = stopwords_text.splitlines()

    return [tk for tk in tokens if tk not in stopwords]


def stemming(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_tokens: list[str] = []
    for tk in tokens:
        stemmed_tokens.append(stemmer.stem(tk))

    return stemmed_tokens


class InvertedIndex:
    type Counter = dict[str, int]

    def __init__(
        self,
        index: dict[str, set[int]],
        docmap: dict[int, dict],
        term_frequencies: dict[int, Counter],
        doc_lengths: dict[int, int],
    ):
        self.index = index
        self.docmap = docmap
        self.term_frequencies = term_frequencies
        self.doc_lengths = doc_lengths

    def __add_document(self, doc_id: int, text: str):
        text_tokens = tokenize_str(text)

        for token in set(text_tokens):
            if self.index.get(token) == None:
                self.index[token] = set()
            self.index[token].add(doc_id)

        for token in text_tokens:
            if self.term_frequencies.get(doc_id) == None:
                self.term_frequencies[doc_id] = {}
            if self.term_frequencies[doc_id].get(token) == None:
                self.term_frequencies[doc_id][token] = 0
            self.term_frequencies[doc_id][token] += 1

        self.doc_lengths[doc_id] = len(text_tokens)

    def __get_avg_doc_length(self) -> float:
        num_of_docs = len(self.doc_lengths)
        if num_of_docs == 0:
            return 0.0

        total_length = 0
        for _doc_id, doc_length in self.doc_lengths.items():
            total_length += doc_length

        return total_length / num_of_docs

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower())
        if doc_ids is not None:
            return sorted(doc_ids)
        return []

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_str(term)
        if len(tokens) != 1:
            raise ValueError("Term includes more than one token.")
        counter = self.term_frequencies.get(doc_id)
        if counter is not None:
            frequency = counter.get(tokens[0])
            if frequency is not None:
                return frequency
        return 0

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1) -> float:
        raw_tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        len_norm = 1 - BM25_B + BM25_B * (doc_length / avg_doc_length)
        bm_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * len_norm)
        return bm_tf

    def get_idf(self, term: str) -> float:
        tokens = tokenize_str(term)
        if len(tokens) != 1:
            raise ValueError("Term includes more than one token.")
        doc_count = len(self.docmap)
        term_doc_count = 0
        if self.index.get(tokens[0]) is not None:
            term_doc_count = len(self.index[tokens[0]])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_str(term)
        if len(tokens) != 1:
            raise ValueError("Term includes more than one token.")
        doc_count = len(self.docmap)
        term_doc_count = 0
        if self.index.get(tokens[0]) is not None:
            term_doc_count = len(self.index[tokens[0]])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int):
        query_tokens = tokenize_str(query)
        scores: dict[int, float] = {}
        for qt in query_tokens:
            if self.index.get(qt) == None:
                break
            for doc_id in self.index[qt]:
                if scores.get(doc_id) == None:
                    scores[doc_id] = 0
                bm25_score = self.bm25(doc_id, qt)
                scores[doc_id] += bm25_score
        docs_by_score = [
            {k: v} for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return docs_by_score[:limit]

    def build(self, movies: dict[str, str]):
        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(
                movie["id"],
                f"{movie['title']} {movie[
                'description'
            ]}",
            )

    def save(self):
        os.makedirs(root_path / "cache", exist_ok=True)

        with open(root_path / "cache" / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(root_path / "cache" / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(root_path / "cache" / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(root_path / "cache" / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(root_path / "cache" / "index.pkl", "rb") as f:
            stored_index = pickle.load(f)
            if stored_index is None:
                raise FileNotFoundError("There is no stored index.")
            else:
                self.index = stored_index

        with open(root_path / "cache" / "docmap.pkl", "rb") as f:
            stored_docmap = pickle.load(f)
            if stored_docmap is None:
                raise FileNotFoundError("There is no stored docmap.")
            else:
                self.docmap = stored_docmap

        with open(root_path / "cache" / "term_frequencies.pkl", "rb") as f:
            stored_term_frequencies = pickle.load(f)
            if stored_term_frequencies is None:
                raise FileNotFoundError("There is no stored term frequencies.")
            else:
                self.term_frequencies = stored_term_frequencies

        with open(root_path / "cache" / "doc_lengths.pkl", "rb") as f:
            stored_doc_lengths = pickle.load(f)
            if stored_doc_lengths is None:
                raise FileNotFoundError("There is no stored term frequencies.")
            else:
                self.doc_lengths = stored_doc_lengths
