import os
import pandas as pd

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader



train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")


train_samples = [
    InputExample(texts=[row["query_text"], row["answer_text"]], label=float(row["relevance"]))
    for _, row in train_df.iterrows()
]

folder_path = r"C:\Users\filip\Projects\Neural_IR_Expansion\hpo"
experiment_name = "crossencoder_training"
destination_path = os.path.join(folder_path, experiment_name) + ".txt"

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

model.fit(
    train_dataloader=train_dataloader,
    epochs=3,
    warmup_steps=100,
    output_path=destination_path
)