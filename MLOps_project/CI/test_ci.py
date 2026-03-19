from src.modello import sentiment_task 
from src.dataset import *

class TestClass :
    """
    test sul modello per pipeline CI
    """
    
    def check_trivial_output(self) :
        assert sentiment_task("neutral")[0]["label"]=="neutral" and sentiment_task("awesome")[0]["label"]=="positive" and sentiment_task("terrible")[0]["label"]=="negative"

    def train_set_bigger_than_test_set(self) :
        assert df_train.shape[0]>df_test.shape[0]