# Utilities
from src.modello import Modello 
from src.dataset import LoadDataset
from sklearn.metrics import accuracy_score

model = Modello()
ld = LoadDataset()

class TestClass :
    """
    unit test sul modello per pipeline CI
    """ 
    
    def test_trivial_output(self) :
        # Controllo del funzionamento del modello con frasi banali
        assert model.predict("neutral")[0]=="neutral" and model.predict("awesome")[0]=="positive" and model.predict("terrible")[0]=="negative"
    
    def test_accuracy(self) :
        # Controllo che l'accuracy sia almeno 0.5
        X = ld.dataset.X
        y = ld.dataset.y
        y_pred = model.predict(X)
        assert accuracy_score(y, y_pred)>=0.5
