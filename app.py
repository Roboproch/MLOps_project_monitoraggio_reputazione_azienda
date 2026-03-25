# Utilities
from src.modello import Modello
from src.dataset import LoadDataset
from sklearn.metrics import accuracy_score
import gradio as gr

model = Modello()
dataset = LoadDataset()

X = dataset.X
y = dataset.y
y_pred = model.predict(X)

valori = ['negative', 'neutral', 'positive']

def predict_tweet(tweet,esito_atteso) :
    X.append(tweet)
    y.append(esito_atteso)
    y_new = model.predict(tweet)[0]
    y_pred.append(y_new)
    acc_new = f"{accuracy_score(y, y_pred)}"
    return y_new,acc_new

demo = gr.Interface(
    fn=predict_tweet,
    inputs=[gr.Textbox(label="Tweet"),gr.Dropdown(choices=valori,label="Esito atteso")],
    outputs=[gr.Textbox(label="Previsione modello"),gr.Textbox(label="Ricalcolo accuracy")],
    flagging_mode = 'never'
)

if __name__ == "__main__":
    demo.launch()
