import re
import numpy as np
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# Preprocess text for BM25
def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return tokens


def mean_ndcg(all_scores, all_true_relevance, k=10):
    ndcgs = []
    for scores, rels in zip(all_scores, all_true_relevance):
        ndcg = ndcg_score([rels], [scores], k=k)
        ndcgs.append(ndcg)
    return np.mean(ndcgs)


def mean_map(all_scores, all_true_relevance):
    maps = []
    for scores, rels in zip(all_scores, all_true_relevance):
        score = label_ranking_average_precision_score([rels], [scores])
        maps.append(score)
    return np.mean(maps)


def mean_precisionk(all_scores, all_true_relevance, k=10):
    precisions = []
    for scores, rels in zip(all_scores, all_true_relevance):
        sorted_idx = np.argsort(scores)[::-1][:k]
        sorted_rels = np.array(rels)[sorted_idx]
        precision = np.sum(sorted_rels) / k
        precisions.append(precision)
    return np.mean(precisions)


def mean_recallk(all_scores, all_true_relevance, k=10):
    recalls = []
    for scores, rels in zip(all_scores, all_true_relevance):
        total_relevant = np.sum(rels)
        if total_relevant == 0:
            recalls.append(0.0)
        else:
            sorted_idx = np.argsort(scores)[::-1][:k]
            sorted_rels = np.array(rels)[sorted_idx]
            recall = np.sum(sorted_rels) / total_relevant
            recalls.append(recall)
    return np.mean(recalls)


# Compute and return all metrics
def evaluate_metrics(scores, cont_rel, bin_rel):
    ndcg = mean_ndcg(scores, cont_rel)
    map = mean_map(scores, bin_rel)
    precision = mean_precisionk(scores, bin_rel)
    recall = mean_recallk(scores, bin_rel)

    return ndcg, map, precision, recall