# Utilities
from src.modello import Modello
from src.dataset import LoadDataset
from sklearn.metrics import accuracy_score
import gradio as gr

model = Modello()
dataset = LoadDataset()

X = dataset.X
y = dataset.y

def predict(data) :
    return model.predict(data)

demo = gr.Interface(
    fn=predict,
    inputs='text_in',
    outputs='text_out'
)

demo.launch()

#y_pred=model.predict(X)

#check_loop = True

#while check_loop :
#   tweet = input("Inserire tweet:")
#  if tweet=="" :
#        print("EXIT")
#        check_loop = False
#    else :
#        print(f"Sentiment: {model.predict(tweet)[0]}")
