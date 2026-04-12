from transformers import pipeline
import torch
import pandas as pd
import numpy as np

print("CUDA available:", torch.cuda.is_available())
print("Num GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

# Use GPU if available
if torch.cuda.is_available():
    device = 0
else:
    device = -1

# Load FinBERT sentiment model
sentiment = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    device=device
)

# Convert column to list (faster for pipeline)
for i in range(2, 11):
    print('Now working on part', i)
    dataset = pd.read_csv(f'data/external_data_part_{i}.csv')
    texts = dataset["Article_title"].fillna("").tolist()

    # Run batch inference
    results = sentiment(
        texts,
        batch_size=512,   # adjust depending on GPU memory
        truncation=True
    )

    # Convert to dataframe
    sentiment_df = pd.DataFrame(results)

    dataset["sentiment_label"] = sentiment_df["label"]
    dataset["sentiment_score"] = sentiment_df["score"]

    # Convert to numeric sentiment
    dataset["sentiment_numeric"] = np.where(
        dataset["sentiment_label"] == "POSITIVE",
        dataset["sentiment_score"],
        np.where(dataset["sentiment_label"] == "NEGATIVE",
                -dataset["sentiment_score"], 0)
    )
    dataset.to_csv(f"data/sentiment_scores/sentiment_scores_{i}.csv", index=False)