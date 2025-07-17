
# ANTIQUE: A Non-Factoid Question Answering Benchmark

ANTIQUE is a non-factoid question answering benchmark based on the questions and answers of Yahoo! Webscope L6. In the following, you can find a brief summary of the dataset.

We conducted the following steps for pre-processing:
1. we kept the non-factoid questions of the dataset (which is called the nfL6 dataset)
2. questions with less than 3 terms were omitted (excluding punctuation marks)
3. questions with no user selected answer were removed
4. duplicate or near-duplicate questions were automatically identified and removed
5. the questions under the categories of "Yahoo! Products" and "Computers & Internet" were omitted. 

From the remaining non-factoid questions, we randomly sampled 2,626 questions (out of 66,634), and dedicated 2426 questions to the training set, and 200 questions to the test set. 

The dataset contains four files, including "antique-collection.txt", "antique-train-queries.txt", "antique-test-queries.txt", "antique-train.qrel", and "antique-test.qrel".

The "antique-collection.txt" file contains all candidate answers that should be considered for answer retrieval. It consists of all human written answers for 66,634 non-factoid questions that we obtained after pre-processing. The file format is tab-separated as follows:
- CandidateAnswerId1	AnswerText1, CandidateAnswerId2	AnswerText2

The CandidateAnswerId format is x_y, where x and y respectively denote the question id and the candidate answer id for the corresponding question. For instance, 2146313_8 shows the 9th candidate question for the question 2146313 (note that the candidate ids start from 0).

The files "antique-train-queries.txt" and "antique-test-queries.txt" respectively contain the question texts for the train and test splits, with the following tab-separated format:
QuestionId1	QuestionText1
QuestionId2	QuestionText2

"antique-train.qrel" and "antique-test.qrel" consist of the relevance labels for the train and test queries. We asked 3 Amazon workers to judge each question-answer pair in a scale ranged from 1 to 4. For definition of each label, we refer the reader to the dataset paper (see below). The format of these two files is as follows:
QuestionId1	Q0\U0\E0	CandidateAnswerId1	RelevanceJudgement
QuestionId2	Q0\U0\E0	CandidateAnswerId2	RelevanceJudgement

where Q0 means that this relevance label was collected via crowdsourcing; E0 means that the crowdworkers did not agree and we could not aggregate their inputs in two rounds of crowdsourcing. Therefore, an expert judge those question-answer pairs. U0 means that the answer was chosen by the asker (author of the question) as the correct answer. We respect the user's explicit feedback and the relevance label for all of these question-answer pairs is set to 4. We believe this discrimination facilitates further research.

Note that the test relevance labels were obtained through depth-k pooling (k=10). We collected 27,422 annotations for training, and 6,589 annotations for testing. The statistics of each label, and the details of the crowdsourcing process can be found in the paper.

To compute the evaluation metrics on this dataset, we recommend the following setting:
1. For the metrics with binary labels (e.g., MAP, MRR, Precision@k, Recall@k, etc.): Assume that labels 1 and 2 are non-relevant, and labels 3 and 4 are relevant.
2. For the metrics with graded relevance labels (e.g., NDCG): Map the 1-4 scale to the 0-3 scale (i.e., subtract all the labels by 1).

Note: The dataset may contain offensive (and noisy) questions and answers. We intentionally did not remove such queries to preserve the nature of the open-domain CQA data. However, if you prefer to omit these questions from your evaluation set, you can use the 'test-queries-blacklist.txt' file and remove the mentioned query IDs from the test set.

ANTIQUE is publicly available for research purposes. If you find this dataset useful, please cite the following article:

Helia Hashemi, Mohammad Aliannejadi, Hamed Zamani, W. Bruce Croft: ANTIQUE: A Non-Factoid Question Answering Benchmark. CoRR abs/1905.08957 (2019).

Or use the following bibtex:

@article{Hashemi:antique:2019, 
author = {Helia Hashemi and Mohammad Aliannejadi and Hamed Zamani and W. Bruce Croft}, 
title = {{ANTIQUE:} {A} Non-Factoid Question Answering Benchmark}, 
journal = {CoRR}, 
volume = {abs/1905.08957}, 
year = {2019}, 
url = {http://arxiv.org/abs/1905.08957}
}
