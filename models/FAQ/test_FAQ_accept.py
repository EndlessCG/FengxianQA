from tqdm import tqdm
from utils.operations import load_faq_questions, load_sim_questions
from runners.faq_runner import FAQRunner
from config import faq_runner_config
import numpy as np

ACCEPT_THRESHOLDS = np.arange(0.1, 0.9, 0.1)
def main():
    model = FAQRunner(faq_runner_config)
    refuse, accept = [], []
    sim_problems = load_sim_questions("input/data/fengxian/sim/test.txt")
    faq_problems = load_faq_questions("input/data/fengxian/faq/test_data")
    for threshold in ACCEPT_THRESHOLDS:
        for sim_problem in tqdm(sim_problems):
            _, confidence = model.do_qa(sim_problem)
            refuse.append(1 if confidence < threshold else 0)
        print("Threshold:", threshold, "Refuse accuracy:", sum(refuse) / len(refuse))
        for faq_problem in tqdm(faq_problems):
            _, confidence = model.do_qa(faq_problem)
            accept.append(1 if confidence > threshold else 0)
        print("Threshold:", threshold, "Accept accuracy:", sum(accept) / len(accept))

if __name__ == '__main__':
    main()