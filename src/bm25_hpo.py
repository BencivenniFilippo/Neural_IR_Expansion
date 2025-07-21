import os
import re
import pandas as pd
import optuna
from sklearn.metrics import ndcg_score
from rank_bm25 import BM25Okapi
from contextlib import redirect_stdout
import numpy as np

from utils import preprocess


train_df = pd.read_csv("data/MyData/train.csv")

def bm25_scores(query_tokens, answers_tokens, k1=0.7, b=0.3):
    bm25 = BM25Okapi(answers_tokens, k1=k1, b=b)
    scores = bm25.get_scores(query_tokens)
    return scores

def evaluate_bm25(df, k1, b):
    ndcgs = []

    for query_id, group in df.groupby("query_id"):
        query = group["query_text"].iloc[0]
        query_input = preprocess(query)

        answers = group["answer_text"]
        answers_input = [preprocess(ans) for ans in answers]

        true_relevance = group["relevance"].tolist()

        scores = bm25_scores(query_input, answers_input, k1, b).tolist()
        ndcgs.append(ndcg_score([true_relevance], [scores], k=10))

    return np.mean(ndcgs)

def objective(trial):
    k1 = trial.suggest_float("k1", 0.5, 2.0)
    b = trial.suggest_float("b", 0.3, 1.0)
    
    return evaluate_bm25(train_df, k1, b)


if __name__ == "__main__":
    # Run the study
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    # Save the results in a .txt file
    folder_path = r"C:\Users\filip\Projects\Neural_IR_Expansion\results"
    experiment_name = "hpo_bm25_10"
    destination_path = os.path.join(folder_path, experiment_name) + ".txt"
    
    with open(destination_path, "w") as f:
        with redirect_stdout(f):
            study.optimize(objective, n_trials=50, show_progress_bar=True)
            f.write(f"Best hyperparameters: {study.best_params}\n")
            f.write(f"Best NDCG score: {study.best_value}\n")
    
    print("Best hyperparameters:", study.best_params)
    print("Best NDCG score:", study.best_value)