import os
import pandas as pd

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


if __name__ == "__main__":
    train_df = pd.read_csv("data/MyData/train.csv") # Use the training dataset

    # Obtain the required format for the input data
    train_samples = [
        InputExample(texts=[row["query_text"], row["answer_text"]], label=float(row["relevance"]))
        for _, row in train_df.iterrows()
    ]
    
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)
    
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
    
    model.fit(
        train_dataloader=train_dataloader,
        epochs=10,
        warmup_steps=100,
    )

    folder_path = "models"
    experiment_name = "crossencoder_fineTuning"
    destination_path = os.path.join(folder_path, experiment_name)
    os.makedirs(destination_path, exist_ok=True)

    model.save(destination_path) # Save the model