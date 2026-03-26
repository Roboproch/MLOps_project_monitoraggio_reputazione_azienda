from datasets import load_dataset
import pandas as pd

class LoadDataset :
    """
    Classe utilizzata per scaricare il dataset e rendere disponibili set di train e set di test
    """

    def __init__(self) :
        # Import del dataset da Hugging Face
        # Prendo solo 200 record per problemi di timeout in fase di caricamento su HuggingFace di un dataset troppo grosso
        self.ds = load_dataset("SetFit/tweet_sentiment_extraction", split="train", streaming=True).take(10)

        # Per comodità trasformo il dataset in un dataframe di pandas
        # df = ds.to_pandas()
        self.df = pd.DataFrame(list(self.ds))

        # Identifico feature e target
        self.X = self.df['text'].tolist()
        self.y = self.df['label_text'].tolist()

        # Considero solo una parte del dataset per motivi prestazionali
        #X = X.tolist()
        #y = y.tolist()
