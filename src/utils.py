import re
import numpy as np
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return tokens


def mean_ndcg(all_scores, all_true_relevance, k=None):
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


def mean_precision(all_scores, all_true_relevance):
    precisions = []
    for scores, rels in zip(all_scores, all_true_relevance):
        # Sort indices descending by score
        sorted_idx = np.argsort(scores)[::-1]
        sorted_rels = np.array(rels)[sorted_idx]
        precision = np.sum(sorted_rels) / len(sorted_rels)  # relevant / retrieved (all)
        precisions.append(precision)
    return np.mean(precisions)


def mean_recall(all_scores, all_true_relevance):
    recalls = []
    for scores, rels in zip(all_scores, all_true_relevance):
        sorted_idx = np.argsort(scores)[::-1]
        sorted_rels = np.array(rels)[sorted_idx]
        total_relevant = np.sum(rels)
        if total_relevant == 0:
            recalls.append(0.0)
        else:
            recall = np.sum(sorted_rels) / total_relevant  # relevant retrieved / total relevant
            recalls.append(recall)
    return np.mean(recalls)


def evaluate_metrics(scores, cont_rel, bin_rel):
    ndcg = mean_ndcg(scores, cont_rel)
    map = mean_map(scores, bin_rel)
    precision = mean_precision(scores, bin_rel)
    recall = mean_recall(scores, bin_rel)

    return ndcg, map, precision, recall


def load_model(model_id):
    save_directory = "./phi2-onnx"  # Local directory to save

    # Load and convert
    model = ORTModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Save the ONNX model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

#load_model("microsoft/phi-2")