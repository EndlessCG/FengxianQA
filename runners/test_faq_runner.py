from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from utils import load_faq_questions, load_sim_questions, merge_arg_and_config, convert_config_paths
from .faq_runner import FAQRunner
from config import faq_runner_config
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept_thresholds",
                        default=np.arange(0.1, 0.9, 0.1))
    parser.add_argument("--sim_test_file", default="input/data/sim/test.txt")
    parser.add_argument("--faq_test_file", default="input/data/faq/test_data")
    parser.add_argument("--n_mixed_inputs", default=500)
    parser.add_argument("--test_types", default=["refuse-acc", "accept-acc", "mixed-faq-acc"])
    parser.add_argument("--refuse_judgement", default="class", choices=["class", "threshold"])
    parser.add_argument("--print_probs", default=False)
    config = faq_runner_config.get("test", dict())
    args = parser.parse_args()
    convert_config_paths(config)
    merge_arg_and_config(args, config)
    return args


def main():
    args = parse_args()
    accept_thresholds = args.accept_thresholds
    n_mixed_inputs = args.n_mixed_inputs

    model = FAQRunner(faq_runner_config)
    sim_problems = load_sim_questions(args.sim_test_file)
    faq_problems = load_faq_questions(args.faq_test_file, get_answers=True)
    min_len = min(len(sim_problems), len(faq_problems))
    mixed_problems = sim_problems[:min(n_mixed_inputs//2, min_len - 1)] + \
                     faq_problems[:min(n_mixed_inputs//2, min_len - 1)]
    np.random.shuffle(mixed_problems)
    result_dict = {}

    for threshold in accept_thresholds:
        if "refuse-acc" in args.test_types:
            results = []
            confidences = []
            for sim_problem in tqdm(sim_problems):
                if args.refuse_judgement == 'class':
                    answer, _ = model.do_qa(sim_problem)
                    result = 1 if answer == "_NO_FAQ_ANSWER" else 0
                else:
                    answer, confidence = model.do_qa(sim_problem)
                    result = 1 if confidence < threshold else 0
                    confidences.append(confidence)
                results.append(result)
            print("Threshold:", threshold, "Refuse-or-not accuracy:",
                  sum(results) / len(results))
            if args.print_probs:
                print("Actual confidences:", confidences)
            result_dict.setdefault("refuse-acc", []).append(sum(results) / len(results))

        if "accept-acc" in args.test_types:
            results = []
            confidences = []
            for faq_problem in tqdm(faq_problems):
                if args.refuse_judgement == 'class':
                    answer, _ = model.do_qa(faq_problem[0])
                    result = 1 if answer != "_NO_FAQ_ANSWER" else 0
                else:
                    answer, confidence = model.do_qa(faq_problem[0])
                    result = 1 if confidence < threshold else 0
                    confidences.append(confidence)
                results.append(result)
            print("Threshold:", threshold, "Accept-or-not accuracy:",
                  sum(results) / len(results))
            if args.print_probs:
                print("Actual confidences:", confidences)
            result_dict.setdefault("accept-acc", []).append(sum(results) / len(results))

        if "mixed-faq-acc" in args.test_types:
            results, labels = [], []
            print("Test size:", len(mixed_problems))
            for mixed_problem in tqdm(mixed_problems):
                if isinstance(mixed_problem, str):
                    # kbqa problem
                    labels.append("-1")
                    pred_id, _, _ = model.do_qa(mixed_problem, get_id=True)
                else:
                    # faq problem
                    labels.append(mixed_problem[1])
                    pred_id, _, _ = model.do_qa(mixed_problem[0], get_id=True)
                results.append(pred_id)
            print("Threshold:", threshold, "Mixed FAQ accuracy:", sum([1 if results[i] == labels[i] else 0 for i in range(len(results))]) / len(results))
            result_dict.setdefault("mixed-faq-acc", []).append(sum([1 if results[i] == labels[i] else 0 for i in range(len(results))]) / len(results))
    print(result_dict)


if __name__ == '__main__':
    main()