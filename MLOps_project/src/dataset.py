from datasets import load_dataset
import pandas as pd

class LoadDataset :
    """
    Classe utilizzata per scaricare il dataset e rendere disponibili set di train e set di test
    """

    # Import del da Hugging Face
    ds = load_dataset("SetFit/tweet_sentiment_extraction")

    # Per comodità trasformo il dataset in un dataframe di pandas
    df_train = ds['train'].to_pandas()
    df_test = ds['test'].to_pandas()
    df = pd.concat([df_train,df_test])

    # Identifico feature e target
    X = df['text'].values
    y = df['label_text'].values
