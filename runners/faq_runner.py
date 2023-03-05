import tensorflow as tf
from models.FAQ.FAQ_model import FAQ
import argparse

class FAQRunner():
    def __init__(self, config):
        self.args = argparse.Namespace(**config)
        model_path = config.get("model_path", "")
        model_dir = config.get("model_dir", "")
        self.model = FAQ(self.args)
        self.model.load_model(model_path=model_path, model_dir=model_dir)

    def disable_warnings(self):
        tf.logging.set_verbosity(tf.logging.ERROR)

    def do_qa(self, sentence):
        return self.model.predict(self.args, sentence)
    
    def __del__(self):
        self.model.destroy()
        