from .bert_kbqa_runner import BertKBQARunner
from config import kbqa_runner_config
from utils import load_sim_questions, load_ner_questions, get_all_sims
from tqdm import tqdm

runner = BertKBQARunner(kbqa_runner_config)
sim_questions = load_sim_questions('input/data/sim/test.txt', get_answers=True)
ner_questions = load_ner_questions('input/data/ner/test.txt', get_answers=True)
assert len(sim_questions) == len(ner_questions)

def main():
    correct_cnt = 0
    all_sims = get_all_sims('input/data/sim/train.txt')
    for i in tqdm(range(len(sim_questions))):
        question, sim_answer = sim_questions[i]
        question_, ner_answer = ner_questions[i]
        assert question == question_
        ner_pred = runner.get_entity(question, get_label_list=True)
        sim_pred = runner.semantic_matching(question, all_sims, 128, get_sgraph=True)
        if ner_pred == ner_answer and sim_pred == sim_answer:
            correct_cnt += 1
    kbqa_accuracy = correct_cnt / len(sim_questions)
    print("KBQA accuracy", kbqa_accuracy)
        

if __name__ == '__main__':
    main()