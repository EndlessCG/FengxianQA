import copy
from tqdm import tqdm
from preprocess.utils import get_question_descriptions
from preprocess.FAQ.back_translation import back_trans
from preprocess.utils import do_request
import random
import torch
import os.path as osp

NER_BASE = "input/data/ner"
SIM_BASE = "input/data/sim"
EL_BASE = "input/data/el"

def generate_data(trans_dests=[]):
    question_descriptions = get_question_descriptions()
    data = []
    entity_synonyms_dict = {}
    for q_desc in tqdm(question_descriptions):
        if len(q_desc) == 6:
            q_type, chart, raw_questions, question_slots, ner, sgraph = q_desc
            repeat = 1
        elif len(q_desc) == 7:
            q_type, chart, raw_questions, question_slots, ner, sgraph, repeat = q_desc
        # question
        question_slot_values = chart[question_slots]
        for raw_question in tqdm(raw_questions):
            for i, val in enumerate(question_slot_values.values):
                temp_data = {}
                temp_data['question'] = raw_question.format(*val) 
                temp_ner, temp_el = [], []
                for key, label in ner:
                    temp_ner.append((chart[key][i], label))
                    # get entity synonyms for EL
                    if chart[key][i] in entity_synonyms_dict:
                        print(f"Already generated synonyms of {chart[key][i]}, using {entity_synonyms_dict[chart[key][i]]}")
                        synonyms = entity_synonyms_dict[chart[key][i]]
                    else:
                        synonyms = do_request(chart[key][i], 5, input_type='word', full_sentence=temp_data['question'])
                        entity_synonyms_dict[chart[key][i]] = synonyms
                    temp_el.append([raw_question, chart[key][i], raw_question.format(*val), synonyms])
                temp_data['qtype'] = q_type
                temp_data['ner'] = temp_ner
                temp_data['sim'] = sgraph
                temp_data['el'] = temp_el
                temp_data["repeat"] = repeat
                if temp_data not in data:
                    data.append(temp_data)
                
                # insert back translation data
                for trans_dest in trans_dests:
                    btrans_data = copy.deepcopy(temp_data)
                    btrans_data['question'] = back_trans(temp_data['question'], trans_dest)
                    if btrans_data['question'] != "" and btrans_data not in data:
                        data.append(btrans_data) 
    return data

def split_data(data, rseed=2023, ratio=[0.6, 0.2, 0.2], shuffle=True):
    assert sum(ratio) == 1
    if shuffle:
        random.seed(rseed) # fix seed to make results reproducible
        random.shuffle(data)
    acc = 0
    datasets = []
    for r in ratio:
        start_idx = acc
        end_idx = start_idx + round(r * len(data))
        datasets.append(data[start_idx:end_idx])
        acc = end_idx
    return datasets

def make_sim_dataset(data, all_sims, neg_to_pos=3):
    filled_data = []
    for pos_data in copy.deepcopy(data):
        for _ in range(pos_data["repeat"]):
            pos_data["label"] = 1
            filled_data.append(pos_data)
            
            if neg_to_pos == -1:
                neg_sample = copy.deepcopy(all_sims)
                neg_sample.remove(pos_data["sim"])
            else:
                neg_sample = random.sample(all_sims, neg_to_pos)
                while pos_data["sim"] in neg_sample:
                    neg_sample = random.sample(all_sims, neg_to_pos)

            for neg_sim in neg_sample:
                # 负例：正例同问题，同问题类型，不同子图和标签
                neg_data = dict()
                neg_data["question"] = pos_data["question"]
                neg_data["label"] = 0
                neg_data["sim"] = neg_sim
                neg_data["qtype"] = pos_data["qtype"]
                filled_data.append(neg_data)
    return filled_data

def make_ner_dataset(data):
    labeled_data = []
    for item in data:
        for _ in range(item["repeat"]):
            entity_labels = item["ner"]
            labels = ['O'] * len(item["question"])
            for entity, label in entity_labels:
                entity_pos = item["question"].find(entity)
                for i in range(entity_pos, entity_pos + len(entity)):
                    assert labels[i] == 'O', f"duplicate labels {item['ner']} in {item['question']}"
                    if label == "entity" and i == entity_pos:
                        labels[i] = "B-entity"
                    elif label == 'entity' and i != entity_pos:
                        labels[i] = 'I-entity'
                    elif label == 'attribute' and i == entity_pos:
                        labels[i] = 'B-attribute'
                    elif label == 'attribute' and i != entity_pos:
                        labels[i] = 'I-attribute'
            labeled_data.append([item["question"], labels, item["qtype"]])
    return labeled_data

def _get_synonyms_series(entity_synonyms, temp_result=[], result_list=[]):
    # DFS
    if len(temp_result) == len(entity_synonyms):
        result_list.append(temp_result)
        return result_list
    for result in entity_synonyms[len(temp_result)]:
        _get_synonyms_series(entity_synonyms, temp_result + [result], result_list)
    return result_list


def make_el_dataset(data, neg_to_pos=1):
    assert neg_to_pos >= 1, "Not supported"
    filled_data = []
    all_entities = []
    
    for item in data:
        for ner in item['ner']:
            all_entities.append(ner[0])
    
    all_entities = list(set(all_entities))
    for item in data:
        for _ in range(item["repeat"]):  
            entity_synonyms = [el[-1] for el in item['el']]
            synonyms_series = _get_synonyms_series(entity_synonyms, [], [])
            for series in synonyms_series:
                for i, mention in enumerate(series):
                    raw_question, entity, original_question, _ = item['el'][i]
                    if mention == entity:
                        filled_data.append([item["qtype"], mention, entity, raw_question.format(*series), 2, original_question])
                    else:
                        filled_data.append([item["qtype"], mention, entity, raw_question.format(*series), 1, original_question])
                    neg_entities = random.sample(all_entities, neg_to_pos)
                    for neg_entity in neg_entities:
                        while neg_entity == entity:
                            neg_entity = random.sample(all_entities, 1)[0]
                            if neg_entity in neg_entities:
                                continue
                        filled_data.append([item["qtype"], mention, neg_entity, raw_question.format(*series), 0, original_question])
    return filled_data

def write_ner_dataset(train, dev, test, base_dir, split_test=True):
    ftrain, fdev, ftest = [open(osp.join(base_dir, path), 'w+') for path in ['train.txt', 'validate.txt', 'test.txt']]
    for question, label, _ in train:
        for char_q, char_l in zip(question, label):
            ftrain.write(f"{char_q} {char_l}\n")
        ftrain.write("\n")
    for question, label, _ in dev:
        for char_q, char_l in zip(question, label):
            fdev.write(f"{char_q} {char_l}\n")
        fdev.write("\n")
    for question, label, _ in test:
        for char_q, char_l in zip(question, label):
            ftest.write(f"{char_q} {char_l}\n")
        ftest.write("\n")
    ftrain.close()
    fdev.close()
    ftest.close()
    if split_test:
        f1hop, fmhop, funchain1hop, funchainmhop = [open(osp.join(base_dir, f"test_{qtype}.txt"), 'w+') for qtype in ['1hop', 'mhop', 'unchain1hop', 'unchainmhop']]
        for question, label, qtype in test:
            if qtype in ["EaT", "EeT"]:
                for char_q, char_l in zip(question, label):
                    f1hop.write(f"{char_q} {char_l}\n")
                f1hop.write("\n")
            elif qtype in ["TaA", "TeE"]:
                for char_q, char_l in zip(question, label):
                    funchain1hop.write(f"{char_q} {char_l}\n")
                funchain1hop.write("\n")
            elif qtype in ["EeNaT", "EeNeT"]:
                for char_q, char_l in zip(question, label):
                    fmhop.write(f"{char_q} {char_l}\n")
                fmhop.write("\n")
            elif qtype in ["EeTaA", "EeTeE", "TeNaA", "TeNeE"]:
                for char_q, char_l in zip(question, label):
                    funchainmhop.write(f"{char_q} {char_l}\n")
                funchainmhop.write("\n")



def write_sim_dataset(train, dev, test, base_dir, split_test=True):
    ftrain, fdev, ftest = [open(osp.join(base_dir, path), 'w+') for path in ['train.txt', 'validate.txt', 'test.txt']]
    for i, item in enumerate(train):
        ftrain.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
    for i, item in enumerate(dev):
        fdev.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
    for i, item in enumerate(test):
        ftest.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
    ftrain.close()
    fdev.close()
    ftest.close()
    if split_test:
        f1hop, fmhop, funchain1hop, funchainmhop = [open(osp.join(base_dir, f"test_{qtype}.txt"), 'w+') for qtype in ['1hop', 'mhop', 'unchain1hop', 'unchainmhop']]
        for i, item in enumerate(test):
            qtype = item["qtype"]
            if qtype in ["EaT", "EeT"]:
                f1hop.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
            elif qtype in ["TaA", "TeE"]:
                funchain1hop.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
            elif qtype in ["EeNaT", "EeNeT"]:
                fmhop.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
            elif qtype in ["EeTaA", "EeTeE", "TeNaA", "TeNeE"]:
                funchainmhop.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))

def write_el_dataset(train, dev, test, base_dir, split_test=True):
    ftrain, fdev, ftest = [open(osp.join(base_dir, path), 'w+') for path in ['train.txt', 'validate.txt', 'test.txt']]
    for item in train:
        ftrain.write("{}\t{}\t{}\t{}\t{}\n".format(*item[1:]))
    for item in dev:
        fdev.write("{}\t{}\t{}\t{}\t{}\n".format(*item[1:]))
    for item in test:
        ftest.write("{}\t{}\t{}\t{}\t{}\n".format(*item[1:]))
    ftrain.close()
    fdev.close()
    ftest.close()
    if split_test:
        f1hop, fmhop, funchain1hop, funchainmhop = [open(osp.join(base_dir, f"test_{qtype}.txt"), 'w+') for qtype in ['1hop', 'mhop', 'unchain1hop', 'unchainmhop']]
        for item in test:
            qtype = item[0]
            if qtype in ["EaT", "EeT"]:
                f1hop.write("{}\t{}\t{}\t{}\t{}\n".format(*item[1:]))
            elif qtype in ["TaA", "TeE"]:
                funchain1hop.write("{}\t{}\t{}\t{}\t{}\n".format(*item[1:]))
            elif qtype in ["EeNaT", "EeNeT"]:
                fmhop.write("{}\t{}\t{}\t{}\t{}\n".format(*item[1:]))
            elif qtype in ["EeTaA", "EeTeE", "TeNaA", "TeNeE"]:
                funchainmhop.write("{}\t{}\t{}\t{}\t{}\n".format(*item[1:]))

def main():
    # data = generate_data(trans_dests=[])
    # torch.save(data, 'input/data/temp_data.bin')
    data = torch.load('/home/gyc/bert-kbqa-test/bert-kbqa/input/data/temp_data.bin')
    train_data, dev_data, test_data = split_data(data, rseed=202302, ratio=[0.6, 0.2, 0.2], shuffle=True)
    ner_train_set, ner_dev_set, ner_test_set = make_ner_dataset(train_data), make_ner_dataset(dev_data), make_ner_dataset(test_data)
    all_sims = set([d["sim"] for d in data])
    sim_train_set, sim_dev_set, sim_test_set = make_sim_dataset(train_data, all_sims=all_sims), make_sim_dataset(dev_data, all_sims=all_sims), make_sim_dataset(test_data, all_sims=all_sims, neg_to_pos=-1)
    el_train_set, el_dev_set, el_test_set = make_el_dataset(train_data), make_el_dataset(dev_data, neg_to_pos=20), make_el_dataset(test_data, neg_to_pos=20)
    write_ner_dataset(ner_train_set, ner_dev_set, ner_test_set, NER_BASE)
    write_sim_dataset(sim_train_set, sim_dev_set, sim_test_set, SIM_BASE)
    write_el_dataset(el_train_set, el_dev_set, el_test_set, EL_BASE)

if __name__ == '__main__':
    main()