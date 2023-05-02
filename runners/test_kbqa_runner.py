import random
from utils.operations import load_el_questions
from .bert_kbqa_runner import BertKBQARunner
from config import kbqa_runner_config, neo4j_config
from utils import load_sim_questions, load_ner_questions, get_all_sims
from tqdm import tqdm

runner = BertKBQARunner(kbqa_runner_config, neo4j_config)
sim_questions = load_sim_questions('input/data/sim/test.txt', get_answers=True)
ner_questions = load_ner_questions('input/data/ner/test.txt', get_answers=True)
el_questions = load_el_questions('input/data/el/test.txt')
all_sims = get_all_sims('input/data/sim/train.txt')
assert len(sim_questions) == len(ner_questions)

def test_once(question, ner_answer, sim_answer, el_entity=None):
    pred_entity_list, pred_attribute_list = runner.get_entity(question)
    real_entity_list, real_attribute_list = runner._decode_ner_tags(ner_answer, question)
    sim_pred = runner.semantic_matching(question, all_sims, 128, get_sgraph=True)
        
    ner_correct = pred_entity_list == real_entity_list and pred_attribute_list == real_attribute_list
    sim_correct = sim_pred == sim_answer
    correct = ner_correct and sim_correct
    
    if kbqa_runner_config.get("entity_linking_method", "fuzzy") == "fuzzy":
        el_pred_e, el_pred_a = runner.fuzzy_entity_linking(pred_entity_list, pred_attribute_list, question=question)
        el_correct = el_entity in el_pred_e or el_entity in el_pred_a
        correct = correct and el_correct
    
    return correct

def test_once_el_sample(question, ner_answer, sim_answer, el_entity, el_mention, original_question):
    pred_entity_list, pred_attribute_list = runner.get_entity(question)
    # pred_entity_list, pred_attribute_list = ner_pred
    real_entity_list, real_attribute_list = runner._decode_ner_tags(ner_answer, original_question)
    sim_pred = runner.semantic_matching(question, all_sims, 128, get_sgraph=True)
    el_pred_e, el_pred_a = runner.fuzzy_entity_linking(pred_entity_list, pred_attribute_list, question=question)

    if el_entity in real_entity_list:
        real_entity_list[real_entity_list.index(el_entity)] = el_mention
    if el_entity in real_attribute_list:
        real_attribute_list[real_attribute_list.index(el_entity)] = el_mention
        
    ner_correct = pred_entity_list == real_entity_list and pred_attribute_list == real_attribute_list
    sim_correct = sim_pred == sim_answer
    el_correct = el_entity in el_pred_e or el_entity in el_pred_a
    if not ner_correct:
        return 0
    if not sim_correct:
        return 1 
    if not el_correct:
        return 2
    return -1

def main():
    err_cnts = [0, 0, 0] # ner, sim, el
    correct_cnt = 0
    total_cnt = 0
    for i in tqdm(range(len(sim_questions))):
        question, sim_answer = sim_questions[i]
        _, ner_answer = ner_questions[i]
        
        if kbqa_runner_config.get("entity_linking_method", "fuzzy") == "fuzzy":
            # Do EL test
            if question not in el_questions:
                print(f"No question {question} in EL dataset")
                continue
            if len(el_questions[question]) > 4:
                # randomly select 3 EL questions to accelerate testing
                question_set = random.sample(el_questions[question][1:], 3)
            for el_mention, el_question, el_entity in question_set:
                # ner_answer[0] += el_mention
                total_cnt += 1
                result = test_once_el_sample(el_question, ner_answer, sim_answer, el_entity, el_mention, question)
                if result == -1:
                    correct_cnt += 1
                else:
                    err_cnts[result] += 1

        else:
            el_entity = None
        
        total_cnt += 1
        if test_once(question, ner_answer, sim_answer, el_entity):
            correct_cnt += 1

    kbqa_accuracy = correct_cnt / total_cnt
    print("KBQA accuracy", kbqa_accuracy)
    print(f"In {total_cnt} tests, correct {correct_cnt}, ner error {err_cnts[0]}, sim error {err_cnts[1]}, el error {err_cnts[2]}")
        

if __name__ == '__main__':
    main()