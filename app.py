# Utilities
from src.modello import Modello
from src.dataset import LoadDataset
from sklearn.metrics import accuracy_score
import gradio as gr

# Dichiaro le variabili e costanti che mi serviranno in seguito
model = None
dataset = None
X = None
y = None
y_pred = None
valori = ['negative', 'neutral', 'positive']

def predict_tweet(tweet,esito_atteso) :
    # Aggiunge gli input (tweet ed esito_atteso) e l'esito predetto al dataset; restituisce l'esito predetto e la nuova accuracy calcolata sul dataset aggiornato
    X.append(tweet)
    y.append(esito_atteso)
    y_new = model.predict(tweet)[0]
    y_pred.append(y_new)
    acc_new = f"{accuracy_score(y, y_pred)}"
    return y_new,acc_new

if __name__ == "__main__":
    # Inizializza gli oggetti
    model = Modello()
    dataset = LoadDataset()
    X = dataset.X
    y = dataset.y
    y_pred = model.predict(X)

    # Definisce l'interfaccia grafica di Gradio
    # Prende in input un testo (tweet) e il sentiment predetto (quest'ultimo scelto da un menu a tendina)
    # Restituisce l'esito predetto e la nuova accuracy calcolata sul dataset aggiornato
    demo = gr.Interface(
        fn=predict_tweet,
        inputs=[gr.Textbox(label="Tweet"),gr.Dropdown(choices=valori,label="Esito atteso")],
        outputs=[gr.Textbox(label="Previsione modello"),gr.Textbox(label="Ricalcolo accuracy")],
        flagging_mode = 'never'
    )

    # Lancia l'interfaccia grafica
    demo.launch()
