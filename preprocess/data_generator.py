import copy
from tqdm import tqdm
from preprocess.utils import get_question_descriptions
from preprocess.FAQ.back_translation import back_trans
import random
import os.path as osp

NER_BASE = "input/data/fengxian/ner"
SIM_BASE = "input/data/fengxian/sim"

def generate_data(trans_dests=[]):
    question_descriptions = get_question_descriptions()
    data = []
    for q_type, chart, raw_questions, question_slots, ner, sgraph in tqdm(question_descriptions):
        # question
        question_slot_values = chart[question_slots]
        for raw_question in raw_questions:
            for i, val in enumerate(question_slot_values.values):
                temp_data = {}
                temp_data['question'] = raw_question.format(*val) 
                temp_ner = []
                for key, label in ner:
                    if key[0] != '=':
                        temp_ner.append((chart[key][i], label))
                    else:
                        temp_ner.append((key[1:], label))
                temp_data['qtype'] = q_type
                temp_data['ner'] = temp_ner
                temp_data['sim'] = sgraph
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

def main():
    data = generate_data(trans_dests=[])
    train_data, dev_data, test_data = split_data(data, rseed=202302, ratio=[0.6, 0.2, 0.2], shuffle=True)
    ner_train_set, ner_dev_set, ner_test_set = make_ner_dataset(train_data), make_ner_dataset(dev_data), make_ner_dataset(test_data)
    all_sims = set([d["sim"] for d in data])
    sim_train_set, sim_dev_set, sim_test_set = make_sim_dataset(train_data, all_sims=all_sims), make_sim_dataset(dev_data, all_sims=all_sims), make_sim_dataset(test_data, all_sims=all_sims, neg_to_pos=-1)
    write_ner_dataset(ner_train_set, ner_dev_set, ner_test_set, NER_BASE)
    write_sim_dataset(sim_train_set, sim_dev_set, sim_test_set, SIM_BASE)

if __name__ == '__main__':
    main()