from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from utils import load_faq_questions, load_sim_questions
from .faq_runner import FAQRunner
from config import faq_runner_config

ACCEPT_THRESHOLDS = [faq_runner_config.get("admit_threshold", 0.3)]
N_MIXED_INPUTS = 500
def main():
    model = FAQRunner(faq_runner_config)
    sim_problems = load_sim_questions("input/data/sim/test.txt")
    faq_problems = load_faq_questions("input/data/faq/test_data", get_answers=True)
    mixed_problems = sim_problems[:N_MIXED_INPUTS//2] + faq_problems[:N_MIXED_INPUTS//2]
    np.random.shuffle(mixed_problems)
    result_dict = {}
    for threshold in ACCEPT_THRESHOLDS:
        results = []
        for sim_problem in tqdm(sim_problems):
            _, confidence = model.do_qa(sim_problem)
            results.append(1 if confidence < threshold else 0)
        print("Threshold:", threshold, "Refuse-or-not accuracy:", sum(results) / len(results))
        result_dict.setdefault("refuse-acc", []).append(sum(results) / len(results))

        results = []
        for faq_problem in tqdm(faq_problems):
            _, confidence = model.do_qa(faq_problem[0])
            results.append(1 if confidence > threshold else 0)
        print("Threshold:", threshold, "Accept-or-not accuracy:", sum(results) / len(results))
        result_dict.setdefault("accept-acc", []).append(sum(results) / len(results))
        
        results, labels = [], []
        for mixed_problem in tqdm(mixed_problems):
            if isinstance(mixed_problem, str):
                # kbqa problem
                labels.append(0)
                _, confidence = model.do_qa(mixed_problem)
            else:
                # faq problem
                labels.append(1)
                _, confidence = model.do_qa(mixed_problem[0])
            results.append(1 if confidence > threshold else 0)
        print("Threshold:", threshold, "Mixed accept-or-refuse F1:", f1_score(y_pred=results, y_true=labels))
        result_dict.setdefault("mixed-accept-f1", []).append(f1_score(y_pred=results, y_true=labels))

        results, labels = [], []
        for mixed_problem in tqdm(mixed_problems):
            if isinstance(mixed_problem, str):
                # kbqa problem
                labels.append(-1)
                pred_id, confidence = model.do_qa(mixed_problem, get_id=True)
            else:
                # faq problem
                labels.append(mixed_problem[1])
                pred_id, confidence = model.do_qa(mixed_problem[0], get_id=True)
            results.append(-1 if confidence < threshold else pred_id)
        print("Threshold:", threshold, "Mixed FAQ accuracy:", sum([1 if results[i] == labels[i] else 0 for i in range(len(results))]) / len(results))
        result_dict.setdefault("mixed-faq-acc", []).append(sum([1 if results[i] == labels[i] else 0 for i in range(len(results))]) / len(results))
    print(result_dict)


if __name__ == '__main__':
    main()