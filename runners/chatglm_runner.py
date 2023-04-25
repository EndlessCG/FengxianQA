from transformers import AutoTokenizer, AutoModel
import argparse

class ChatGLMRunner():
    def __init__(self, config):
        self.args = argparse.Namespace(**config)
        self._verbose = config.get("verbose", "False")
        self._use_local_chatglm = config.get("use_local_chatglm", "False")
        self._print("加载ChatGLM模型...")
        self._load_chatglm_model()
    
    def _print(self, *args):
        if self._verbose:
            print("ChatGLM:", *args)
        
    def _load_chatglm_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.repo_name, trust_remote_code=True, local_file_only=self._use_local_chatglm)
        self.model = AutoModel.from_pretrained(self.args.repo_name, trust_remote_code=True).half().cuda()
        self.model.eval()
        self.history = None

    def do_qa(self, question):
        response, _ = self.model.chat(self.tokenizer, question, history=None)
        return response
