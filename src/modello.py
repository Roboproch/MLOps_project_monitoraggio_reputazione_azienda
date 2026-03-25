from transformers import pipeline

class Modello :

    # Import del modello da Hugging Face
    sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def predict(self,tweet) :
        # Metodo per le predizioni
        return self.sentiment_task(tweet)[0]["label"]