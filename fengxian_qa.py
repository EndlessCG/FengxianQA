from runners.bert_kbqa_runner import BertKBQARunner
from runners.faq_runner import FAQRunner
from config import fengxian_qa_config, kbqa_runner_config, faq_runner_config
import time

class FengxianQA:
    def __init__(self):
        init_start = time.time()
        self.kbqa_runner = BertKBQARunner(kbqa_runner_config)
        self.faq_runner = FAQRunner(faq_runner_config)
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
        faq_answer, faq_prob = self.faq_runner.do_qa(question)
        self._print("FAQ信心：", faq_prob)
        if faq_prob > faq_runner_config.get("admit_threshold", 0.8):
            self._print("使用FAQ回答")
            return faq_answer
        else:
            self._print("使用KBQA回答")
            kbqa_answer = self.kbqa_runner.do_qa(question)
            return kbqa_answer

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
