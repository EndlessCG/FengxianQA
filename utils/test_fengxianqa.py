import random
import time
import numpy as np
from tqdm import tqdm

from fengxian_qa import FengxianQA
from .operations import load_sim_questions, load_faq_questions, load_ner_questions, get_all_sims
from config import faq_runner_config

test_ninputs = [500]
FAQ_THRESHOLDS = [faq_runner_config.get("admit_threshold", 0.3)]
def main():
    faq_questions = load_faq_questions("input/data/faq/test_data", get_answers=True)
    sim_questions = load_sim_questions("input/data/sim/test.txt", get_answers=True)
    ner_questions = load_ner_questions("input/data/ner/test.txt", get_answers=True)
    each_length = min(len(sim_questions), len(faq_questions))
    mixed_questions = []
    for q, a in faq_questions[:each_length]:
        mixed_questions.append([q, a])
    for i in range(each_length):
        mixed_questions.append([i])
    random.shuffle(mixed_questions)

    agent = FengxianQA()
    all_sims = get_all_sims("input/data/sim/test.txt")

    for FAQ_THRESHOLD in FAQ_THRESHOLDS:
        kbqa_cnt, faq_cnt, kbqa_correct_cnt, faq_correct_cnt = 0, 0, 0, 0
        for q in tqdm(mixed_questions):
            if len(q) == 1:
                # kbqa questions
                question, sim_answer = sim_questions[q[0]]
                question_, ner_answer = ner_questions[q[0]]
                assert question == question_
                faq_pred, confidence = agent.faq_runner.do_qa(question)
                ner_pred = agent.kbqa_runner.get_entity(question, get_label_list=True)
                sim_pred = agent.kbqa_runner.semantic_matching(question, all_sims, 128, get_sgraph=True)
                if confidence < FAQ_THRESHOLD and ner_pred == ner_answer and sim_pred == sim_answer:
                    kbqa_correct_cnt += 1
                kbqa_cnt += 1
            elif len(q) == 2:
                # faq questions
                question, answer = q
                faq_pred, confidence = agent.faq_runner.do_qa(question, get_id=True)
                if confidence > FAQ_THRESHOLD and faq_pred == answer:
                    faq_correct_cnt += 1
                faq_cnt += 1
        print("Threshold", FAQ_THRESHOLD)
        print("Data count", kbqa_cnt, faq_cnt)
        print("FengxianQA accuracy", (kbqa_correct_cnt + faq_correct_cnt) / (kbqa_cnt + faq_cnt))
        print("FAQ问题准确率", faq_correct_cnt / faq_cnt)
        print("KBQA问题准确率", kbqa_correct_cnt / kbqa_cnt)
if __name__ == '__main__':
    main()