from bert_kbqa_runner import BertKBQARunner
from config import bert_kbqa_config

def main():
    runner = BertKBQARunner(bert_kbqa_config)
    while True:
        print("====="*10)
        raw_text = input("问题：") 
        raw_text = raw_text.strip() # 去掉输入的首尾空格
        if ( "quit" == raw_text ):
            print("quit")
            return
        runner.do_qa(raw_text)

if __name__ == '__main__':
    main()

