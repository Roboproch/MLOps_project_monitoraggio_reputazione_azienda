from ..modello import *

class TestClass :
    """
    test sul modello per pipeline CI
    """
    
    def check_trivial_output(self) :
        assert modello.sentiment_task("neutral")[0]["label"]=="neutral" and modello.sentiment_task("awesome")[0]["label"]=="positive" and modello.sentiment_task("terrible")[0]["label"]=="negative"

    def train_set_bigger_than_test_set(self) :
        assert modello.df_train.shape[0]>modello.df_test.shape[0]