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

def test_once(question, ner_answer, sim_answer, el_entity):
    pred_entity_list, pred_attribute_list = runner.get_entity(question)
    real_entity_list, real_attribute_list = runner._decode_ner_tags(ner_answer, question)
    sim_pred = runner.semantic_matching(question, all_sims, 128, get_sgraph=True)
    el_pred_e, el_pred_a = runner.fuzzy_entity_linking(pred_entity_list, pred_attribute_list, question=question)
        
    ner_correct = pred_entity_list == real_entity_list and pred_attribute_list == real_attribute_list
    sim_correct = sim_pred == sim_answer
    el_correct = el_entity in el_pred_e or el_entity in el_pred_a
    if ner_correct and sim_correct and el_correct:
        return True
    return False

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
    if ner_correct and sim_correct and el_correct:
        return True
    return False

def main():
    correct_cnt = 0
    total_cnt = 0
    for i in tqdm(range(len(sim_questions))):
        question, sim_answer = sim_questions[i]
        _, ner_answer = ner_questions[i]
        _, _, el_entity = el_questions[question][0]
        total_cnt += 1
        if test_once(question, ner_answer, sim_answer, el_entity):
            correct_cnt += 1
        for el_mention, el_question, el_entity in el_questions[question][1:]:
            # ner_answer[0] += el_mention
            total_cnt += 1
            if test_once_el_sample(el_question, ner_answer, sim_answer, el_entity, el_mention, question):
                correct_cnt += 1

    kbqa_accuracy = correct_cnt / total_cnt
    print("KBQA accuracy", kbqa_accuracy)
        

if __name__ == '__main__':
    main()