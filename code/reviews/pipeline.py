"""
pipeline.py — Step 2: call the APIs once, cache what they say, flatten it into tables.

A pipeline built on someone else's API has two kinds of step, and keeping them
apart is the whole design:

- **Extract steps** call the API — slow, rate-limited, and the only part that can
  fail because of the network. Each one saves the raw JSON it got back to a cache
  file, and the next time it runs it reads that file instead of calling again.
  (Lesson 4-2: cache every call you can.)
- **Transform steps** turn that JSON into DataFrames. They never touch the
  network, so you can run them a hundred times while you get them right.

    reviews ─sentiment_step─▶ sentiment JSON ─sentences_table─▶ a row per sentence
                                            └opinions_table──▶ a row per target
    reviews ─entities_step──▶ entities JSON ─entities_table──▶ a row per entity
    mentions ─subjects_step─▶ a row per mention, with its subject

Each arrow changes the **grain** — review, sentence, target, entity — the idea
from Assignment 05, now coming out of JSON instead of a CSV.

Less scaffolding here: the steps are described, the code is yours.
"""

import json
import os

import pandas as pd

from .apicalls import classify_mentions, get_entities, get_sentiment
from .settings import MENTION_CATEGORIES


def sentiment_step(reviews: pd.DataFrame, cache_file: str) -> list:
    """Sentiment for every review, from the cache if it exists, else from the API."""
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    documents = []
    for review in reviews.to_dict("records"):
        response = get_sentiment(review["text"])
        document = response["results"]["documents"][0]
        document["review_id"] = review["review_id"]
        documents.append(document)

    with open(cache_file, "w") as f:
        json.dump(documents, f)
    return documents


def entities_step(reviews: pd.DataFrame, cache_file: str) -> list:
    """Named entities for every review, cached — the same shape as `sentiment_step`."""
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    documents = []
    for review in reviews.to_dict("records"):
        response = get_entities(review["text"])
        document = response["results"]["documents"][0]
        document["review_id"] = review["review_id"]
        documents.append(document)

    with open(cache_file, "w") as f:
        json.dump(documents, f)
    return documents


def sentences_table(documents: list) -> pd.DataFrame:
    """One row per sentence: the sentiment documents flattened with `json_normalize`."""
    df = pd.json_normalize(
        documents,
        record_path="sentences",
        meta=["review_id"],
        record_prefix="sentence_",
    )
    df["sentence_id"] = df.groupby("review_id").cumcount()
    df["sentence_text"] = df["sentence_text"].str.strip()
    return df[[
        "review_id",
        "sentence_id",
        "sentence_text",
        "sentence_sentiment",
        "sentence_offset",
        "sentence_length",
    ]]


def opinions_table(documents: list) -> pd.DataFrame:
    """One row per opinion target, with the words that carry the opinion."""
    rows = []
    for document in documents:
        for sentence_id, sentence in enumerate(document["sentences"]):
            assessments = sentence["assessments"]
            for target in sentence["targets"]:
                words = []
                for relation in target["relations"]:
                    if relation["relationType"] == "assessment":
                        index = int(relation["ref"].split("/")[-1])
                        words.append(assessments[index]["text"])
                rows.append({
                    "review_id": document["review_id"],
                    "sentence_id": sentence_id,
                    "mention": target["text"],
                    "mention_sentiment": target["sentiment"],
                    "assessments": ", ".join(words),
                    "mention_offset": target["offset"],
                    "source": "opinion",
                })
    return pd.DataFrame(rows, columns=[
        "review_id",
        "sentence_id",
        "mention",
        "mention_sentiment",
        "assessments",
        "mention_offset",
        "source",
    ])


def entities_table(documents: list) -> pd.DataFrame:
    """One row per person or product the entity recognizer found."""
    df = pd.json_normalize(
        documents,
        record_path="entities",
        meta=["review_id"],
        record_prefix="entity_",
    )
    df = df[df["entity_category"].isin(MENTION_CATEGORIES)]
    return df[[
        "review_id",
        "entity_text",
        "entity_category",
        "entity_offset",
    ]].reset_index(drop=True)


def subjects_step(reviews: pd.DataFrame, mentions: pd.DataFrame,
                  cache_file: str) -> pd.DataFrame:
    """What each mention is about — service, atmosphere, product, other — cached."""
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    texts = dict(zip(reviews["review_id"], reviews["text"]))

    rows = []
    for review_id, group in mentions.groupby("review_id"):
        mention_list = (
            group.sort_values("mention_offset")["mention"]
            .drop_duplicates()
            .tolist()
        )
        response = classify_mentions(texts[review_id], mention_list)
        for mention in mention_list:
            answer = response["answers"][mention]
            rows.append({
                "review_id": review_id,
                "mention": mention,
                "subject": answer["choice"],
                "subject_confidence": answer["confidence"],
            })

    subjects = pd.DataFrame(
        rows, columns=["review_id", "mention", "subject", "subject_confidence"]
    )
    subjects.to_csv(cache_file, index=False)
    return subjects
