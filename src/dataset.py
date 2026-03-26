from datasets import load_dataset
import pandas as pd

class LoadDataset :
    """
    Classe utilizzata per scaricare il dataset e rendere disponibili set di train e set di test
    """

    # Import del da Hugging Face
    # Prendo solo 200 record per problemi di timeout in fase di caricamento su HuggingFace di un dataset troppo grosso
    ds = load_dataset("SetFit/tweet_sentiment_extraction", split="train", streaming=True).take(10)

    # Per comodità trasformo il dataset in un dataframe di pandas
    df = ds.to_pandas()

    # Identifico feature e target
    X = df['text'].values
    y = df['label_text'].values

    # Considero solo una parte del dataset per motivi prestazionali
    X = X.tolist()
    y = y.tolist()
