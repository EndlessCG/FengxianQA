from transformers import AutoTokenizer, AutoModel
import argparse

class ChatGLMRunner():
    def __init__(self, config):
        self.args = argparse.Namespace(**config)
        self._load_chatglm_model()
    
    def _load_chatglm_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.repo_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.args.repo_name, trust_remote_code=True).half().cuda()
        self.model.eval()
        self.history = None

    def do_qa(self, question):
        response, self.history = self.model.chat(self.tokenizer, question, history=self.history)
        return response
