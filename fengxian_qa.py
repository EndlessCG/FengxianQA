from runners.bert_kbqa_runner import BertKBQARunner
from runners.faq_runner import FAQRunner
from config import bert_kbqa_config, faq_config

class FengxianQA:
    def __init__(self):
        self.kbqa_runner = BertKBQARunner(bert_kbqa_config)
        self.faq_runner = FAQRunner(faq_config)
        self.faq_runner.disable_warnings()

    def do_qa(self, question):
        faq_answer, faq_prob = self.faq_runner.do_qa(question)
        print("FAQ信心：", faq_prob)
        if faq_prob > faq_config.get("admit_threshold", 0.8):
            print("使用FAQ回答")
            return faq_answer
        else:
            print("使用KBQA回答")
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
            print(self.do_qa(question))


if __name__ == '__main__':
    fengxian_qa = FengxianQA()
    fengxian_qa.interact()
