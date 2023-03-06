import random
import time
import numpy as np
from tqdm import tqdm

from fengxian_qa import FengxianQA
from .operations import load_sim_questions, load_faq_questions

test_ninputs = [500]
def main():
    faq_questions = load_faq_questions("input/data/fengxian/faq/train_data")
    sim_questions = load_sim_questions("input/data/fengxian/sim/train.txt")
    times = []

    agent = FengxianQA()

    for t in test_ninputs:
        for question in tqdm(faq_questions[:t]):
            start = time.time()
            _ = agent.do_qa(question)
            end = time.time()
            times.append(end - start)
    print(f"FAQ questions: {np.average(times)} +- {np.std(times)}")
    for t in test_ninputs:
        for question in tqdm(sim_questions[:t]):
            start = time.time()
            _ = agent.do_qa(question)
            end = time.time()
            times.append(end - start)
    print(f"SIM questions: {np.average(times)} +- {np.std(times)}")
    for t in test_ninputs:
        mixed_questions = faq_questions[:t//2] + sim_questions[:t//2]
        random.shuffle(mixed_questions)
        for question in tqdm(mixed_questions):
            start = time.time()
            _ = agent.do_qa(question)
            end = time.time()
            times.append(end - start)
    print(f"Equally mixed questions: {np.average(times)} +- {np.std(times)}")
    

        

if __name__ == '__main__':
    main()