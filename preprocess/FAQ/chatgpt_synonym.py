import os
import openai
import xlrd
import json
import requests
from tqdm import tqdm

QA_PATH = "input/data/faq/file/qa100.xls"

url = "https://api.openai-proxy.com/v1/chat/completions"

def do_request(headers, payload, n_extend):
    all_questions = []
    while(len(all_questions) != n_extend):
        all_questions.clear()
        try:
            raw = requests.request("POST", url, headers=headers, data=payload, timeout=60)
        except requests.ReadTimeout:
            print("Timeout")
            raw = None
        
        if raw is not None:
            response = eval(raw.text)
            print(eval(payload)["content"])
            print(response)
            if response['code'] != 200:
                print(f"Request failed for with {response['code']} {response['message']}")
                return None
            for question in response['data'].split('\n'):
                if question == '':
                    continue
                all_questions.append(question.split(' ')[-1])
        
        if raw is None or all_questions is None or len(all_questions) != n_extend:
            print(f"Retrying with invalid output {all_questions}")
    return all_questions

def get_synonyms(sentence, n_extend):
    all_questions = []
    payload = json.dumps({
        "apiKey": os.getenv("OPENAI_API_KEY"),
        "sessionId": "8d1cb9b0-d535-43a8-b738-4f61b1608579",
        "content": f"原句：{sentence}\n输出{min(n_extend, 10)}个中文同义句，省略原句中的部分信息，尽量缩短句子："
    })
    headers = {
        'Content-Type': 'application/json'
    }
    all_questions.extend(do_request(headers, payload, min(n_extend, 10)))
    n_extend -= 10
    while n_extend > 0:
        payload = json.dumps({
            "apiKey": os.getenv("OPENAI_API_KEY"),
            "sessionId": "8d1cb9b0-d535-43a8-b738-4f61b1608579",
            "content": f"再输出\"{sentence}\"的{min(n_extend, 10)}个中文同义句，省略原句中的部分信息，尽量缩短句子："
        })
        all_questions.extend(do_request(headers, payload, min(n_extend, 10)))
        n_extend -= 10
    return all_questions
    

def main():
    workbook = xlrd.open_workbook(QA_PATH)
    sheet = workbook.sheet_by_index(0)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    all_answers = []
    for row in tqdm(sheet[1:]):
        question = row.value[0]
        all_answers.extend(get_synonyms(question))

if __name__ == '__main__':
    main()