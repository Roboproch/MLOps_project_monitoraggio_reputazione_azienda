import torch
from transformers import pipeline

class Modello :
    """
    Gestisce l'import del modello da Hugging Face e implementa un metodo per la predizione del sentiment
    """

    def __init__(self) :
        # Import del modello da Hugging Face
        self.sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def predict(self,tweets) :
        # Metodo per le predizioni, prende in input una o più stringhe (sottoforma di lista eventualmente)
        # Definisco un batch_size per efficienza
        results = self.sentiment_task(tweets, batch_size=32) 
        return [res["label"] for res in results]
