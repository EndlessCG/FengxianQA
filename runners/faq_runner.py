import tensorflow as tf
from models.FAQ import FAQ
import argparse

class FAQRunner():
    def __init__(self, config):
        self.args = argparse.Namespace(**config)
        self.model_path = config.get("model_path", "")
        self.model_dir = config.get("model_dir", "")
        self._load_faq_model()

    def _load_faq_model(self):
        self.model = FAQ(self.args)
        self.model.load_model(model_path=self.model_path, model_dir=self.model_dir)

    def disable_warnings(self):
        tf.logging.set_verbosity(tf.logging.ERROR)

    def do_qa(self, sentence, get_id=False):
        return self.model.predict(self.args, sentence, get_id=get_id)
    
    def __del__(self):
        self.model.destroy()
        