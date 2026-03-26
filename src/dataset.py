from datasets import load_dataset
import pandas as pd

class LoadDataset :
    """
    Scarica il dataset da Hugging Face e ne rende disponibili feature (tweet) e target (sentiment sottoforma di testo)
    """

    def __init__(self) :
        # Import del dataset da Hugging Face
        # Prendo solo 10 record per problemi prestazionali (timeout) in fase di caricamento su HuggingFace di un dataset troppo grande
        self.ds = load_dataset("SetFit/tweet_sentiment_extraction", split="train", streaming=True).take(10)

        # Per comodità trasformo il dataset in un dataframe di pandas
        #self.df = pd.DataFrame(list(self.ds))

        # Identifico feature e target
        #self.X = self.df['text'].tolist()
        #self.y = self.df['label_text'].tolist()

        self.X = []
        self.y = []
        for example in self.ds:
            self.X.append(example['text'])
            self.y.append(example['label_text'])
