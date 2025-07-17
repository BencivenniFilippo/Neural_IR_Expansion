import os
import pandas as pd
import numpy as np

from utils import preprocess
from bm25_hpo import bm25_scores
from sentence_transformers import CrossEncoder
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score


train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

cross_encoder = CrossEncoder("cross_encoder_finetuned")  # Add the model path
top_k = 10


def process_query(query, answers, true_rels):
    q_tok = preprocess(query)
    a_tok = [preprocess(ans) for ans in answers]

    scores = bm25_scores(q_tok, a_tok).tolist()

    # Baseline evaluation on full list:
    # (we’ll compute nDCG@len(answers), MRR, MAP for BM25)
    ndcg_bm25 = ndcg_score([true_rels], [scores], k=len(answers))
    # MRR:
    try:
        first_rel_idx = next(i for i, r in enumerate(scores) if true_rels[i] > 0)
        mrr_bm25 = 1.0 / (first_rel_idx + 1)
    except StopIteration:
        mrr_bm25 = 0.0
    # MAP:
    map_bm25 = label_ranking_average_precision_score([true_rels], [scores])


    top_k_indices = np.argsort(scores)[-top_k:][::-1] # [:] Since I wanna keep all the scores
    top_answers = [answers[i] for i in top_k_indices]
    top_true_rels = [true_rels[i] for i in top_k_indices]

    # Score each (query, answer) pair
    pairs = [[query, ans] for ans in top_answers]
    re_ranked = cross_encoder.predict(pairs, batch_size=8)

    # Sort by the predicted scores
    reranked_answers = [x for _, x in sorted(zip(re_ranked, top_answers), reverse=True)]

    
    # Evaluation on the top_k subset:
    ndcg_rerank = ndcg_score([top_true_rels], [re_ranked], k=top_k)
    try:
        first_rel_idx2 = next(i for i, r in enumerate(re_ranked) if top_true_rels[i] > 0)
        mrr_rerank = 1.0 / (first_rel_idx2 + 1)
    except StopIteration:
        mrr_rerank = 0.0
    map_rerank = label_ranking_average_precision_score([top_true_rels], [re_ranked])

    return {
        "ndcg_bm25": ndcg_bm25,
        "mrr_bm25": mrr_bm25,
        "map_bm25": map_bm25,
        "ndcg_rerank": ndcg_rerank,
        "mrr_rerank": mrr_rerank,
        "map_rerank": map_rerank,
    }


if __name__ == "__main__":

    for query_id, group in test_df.groupby("query_id"):
        query = group["query_text"].iloc[0]
        answers = group["answer_text"]
        true_relevance = group["relevance"].tolist()

        process_query(query, answers, true_relevance)

