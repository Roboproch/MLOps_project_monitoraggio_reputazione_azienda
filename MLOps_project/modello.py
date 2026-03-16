from transformers import pipeline
sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")
# print(sentiment_task("Covid cases are increasing fast!"))

from datasets import load_dataset
ds = load_dataset("SetFit/tweet_sentiment_extraction")
# il dataset viene splittato da load_dataset
# stampa il primo record del set di train
print(ds['train'][0])
# stampa il primo record del set di test
print(ds['test'][0])