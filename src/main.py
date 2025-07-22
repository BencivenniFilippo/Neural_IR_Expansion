import pandas as pd

from utils import preprocess, evaluate_metrics
from bm25_hpo import bm25_scores
from sentence_transformers import CrossEncoder


test_base_df = pd.read_csv("data/MyData/test.csv")
test_novice_df = pd.read_csv("data/MyData/novice_expansion.csv")
test_expert_df = pd.read_csv("data/MyData/expert_expansion.csv")


# Load our fine-tuned cross-encoder
fine_tuned_model = r"C:\Users\filip\Projects\Neural_IR_Expansion\models\crossencoder_fineTuning"
cross_encoder = CrossEncoder(fine_tuned_model)

# Return the BM25 and cross-encoder scores
def process_query(query, answers):
    q_tok = preprocess(query)
    a_tok = [preprocess(ans) for ans in answers]

    # Token inputs for the BM25
    scores = bm25_scores(q_tok, a_tok).tolist()

    # Natural text input for the cross-encoder
    pairs = [[query, ans] for ans in answers]
    re_ranked = cross_encoder.predict(pairs, batch_size=8)
    re_ranked = re_ranked.tolist()

    return scores, re_ranked

# Compute and store results and relevances for BM25 and cross-encoder for later evaluation
def main_loop(df):
    bm25_res = []
    re_ranker_res = []
    relevances_continuous = []
    relevances_binary = []

    for query_id, group in df.groupby("query_id"):
        query = group["query_text"].iloc[0]
        answers = group["answer_text"]
        true_relevance = group["relevance"].tolist()

        bm25_score, re_ranker_score = process_query(query, answers)

        bm25_res.append(bm25_score)
        re_ranker_res.append(re_ranker_score)
        relevances_continuous.append(true_relevance) # Continuous relevances for nDCG

        true_rel_bin = [0 if rel < 0.5 else 1 for rel in true_relevance] # Threshold continuous values at 0.5
        relevances_binary.append(true_rel_bin) # Binary relevances for MAP, Precision@k and Relevance@K

    return bm25_res, re_ranker_res, relevances_continuous, relevances_binary

if __name__ == "__main__":

    # Run main loop for all test sets
    bm25_base, re_ranker_base, cont_rel_base, bin_rel_base = main_loop(test_base_df)
    print("Base df computed")
    bm25_nov, re_ranker_nov, cont_rel_nov, bin_rel_nov = main_loop(test_novice_df)
    print("Novice df computed")
    bm25_expert, re_ranker_expert, cont_rel_expert, bin_rel_expert = main_loop(test_expert_df)
    print("Expert df computed")

    # Evaluate metrics for all sets and store them in a dictionary
    results = {}

    results["BASE_BM25"] = evaluate_metrics(bm25_base, cont_rel_base, bin_rel_base)
    results["BASE_RERANK"] = evaluate_metrics(re_ranker_base, cont_rel_base, bin_rel_base)
    print("Base df evaluated")

    results["NOVICE_BM25"] = evaluate_metrics(bm25_nov, cont_rel_nov, bin_rel_nov)
    results["NOVICE_RERANK"] = evaluate_metrics(re_ranker_nov, cont_rel_nov, bin_rel_nov)
    print("Novice df evaluated")

    results["EXPERT_BM25"] = evaluate_metrics(bm25_expert, cont_rel_expert, bin_rel_expert)
    results["EXPERT_RERANK"] = evaluate_metrics(re_ranker_expert, cont_rel_expert, bin_rel_expert)
    print("Expert df evaluated")

    # Write results in a txt file
    destination_path = r"C:\Users\filip\Projects\Neural_IR_Expansion\results\results_new.txt"
    with open(destination_path, "w") as f:
        for key, (ndcg, map_, precision, recall) in results.items():
            f.write(f"{key}: NDCG={ndcg:.4f}, MAP={map_:.4f}, Precision@k={precision:.4f}, Recall@k={recall:.4f}\n")
