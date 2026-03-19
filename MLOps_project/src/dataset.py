from datasets import load_dataset
import pandas as pd

# Import di un dataset da Hugging Face
ds = load_dataset("SetFit/tweet_sentiment_extraction")
# Il dataset viene splittato autometicamente in train set e test set da load_dataset
# Per comodità trasformo i dataset in dataframe di pandas
df_train = ds['train'].to_pandas()
df_test = ds['test'].to_pandas()

X_train = df_train['text'].values
y_train = df_train['label_text'].values
X_test = df_test['text'].values
y_test = df_test['label_text'].values