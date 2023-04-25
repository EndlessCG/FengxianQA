from runners.bert_kbqa_runner import BertKBQARunner
from runners.faq_runner import FAQRunner
from runners.chatglm_runner import ChatGLMRunner
from config import fengxian_qa_config, kbqa_runner_config, faq_runner_config, chatglm_runner_config, neo4j_config
from utils import convert_config_paths
import time

class FengxianQA:
    def __init__(self):
        init_start = time.time()
        convert_config_paths(kbqa_runner_config)
        convert_config_paths(faq_runner_config)
        convert_config_paths(chatglm_runner_config)
        self.kbqa_runner = BertKBQARunner(kbqa_runner_config, neo4j_config)
        self.faq_runner = FAQRunner(faq_runner_config)
        self.chatglm_runner = ChatGLMRunner(chatglm_runner_config)
        self.faq_runner.disable_warnings()
        self._verbose = fengxian_qa_config.get("verbose", "False")
        init_end = time.time()
        self._print("初始化用时：", init_end - init_start)

    def _print(self, *args):
        if self._verbose:
            print("FengxianQA:", *args)

    def do_qa(self, question):
        if question == "":
            return "请输入问题"
        faq_id, faq_answer, faq_prob = self.faq_runner.do_qa(question, get_id=True)
        self._print("FAQ结果:", faq_id, "置信度:", faq_prob)
        if faq_answer != "_NO_FAQ_ANSWER":
            self._print("使用FAQ回答")
            return faq_answer
        kbqa_ret, kbqa_answer = self.kbqa_runner.do_qa(question)
        if kbqa_ret == 0:
            self._print("使用KBQA回答")
            return kbqa_answer
        chatglm_answer = self.chatglm_runner.do_qa(question)
        self._print("使用ChatGLM回答")
        return chatglm_answer
        

    def only_kbqa(self, question):
        return self.kbqa_runner.do_qa(question)
    
    def only_chatglm(self, question):
        return self.chatglm_runner.do_qa(question)
    
    def interact(self):
        while True:
            print("====="*10)
            raw_text = input("问题：") 
            question = raw_text.strip() # 去掉输入的首尾空格
            if ( "quit" == question ):
                print("quit")
                return
            print("回答：" + self.do_qa(question))


if __name__ == '__main__':
    fengxian_qa = FengxianQA()
    fengxian_qa.interact()
