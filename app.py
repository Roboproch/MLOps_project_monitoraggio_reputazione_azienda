# Utilities
from src.modello import Modello
from src.dataset import LoadDataset
from sklearn.metrics import accuracy_score

model = Modello()
dataset = LoadDataset()

# Considero solo una parte del dataset per motivi prestazionali
X = dataset.X[:200].tolist()
y = dataset.y[:200].tolist()

y_pred=model.predict(X)

check_loop = True

while check_loop :
    tweet = input("Inserire tweet:")
    if tweet=="" :
        print("EXIT")
        check_loop = False
    else :
        print(f"Sentiment: {model.predict(tweet)[0]}")
