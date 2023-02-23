import copy
from utils import get_question_descriptions
import random
import os.path as osp

NER_BASE = "models/input/data/fengxian/ner"
SIM_BASE = "models/input/data/fengxian/sim_data"

def generate_data():
    question_descriptions = get_question_descriptions()
    data = []
    for chart, raw_questions, question_slots, ner, sgraph in question_descriptions:
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
                temp_data['ner'] = temp_ner
                temp_data['sim'] = sgraph
                data.append(temp_data)
    return data

def split_data(data, rseed=2023, ratio=[0.6, 0.2, 0.2]):
    assert sum(ratio) == 1
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

def make_sim_dataset(data, neg_to_pos=5):
    filled_data = []
    all_sims = set([d["sim"] for d in data])
    for pos_data in data:
        pos_data["label"] = 1
        filled_data.append(pos_data)
        for neg_sim in random.sample(all_sims, neg_to_pos):
            neg_data = copy.deepcopy(pos_data)
            neg_data["label"] = 0
            neg_data["sim"] = neg_sim
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
        labeled_data.append([item["question"], labels])
    return labeled_data

def write_ner_dataset(train, dev, test, base_dir):
    ftrain, fdev, ftest = [open(osp.join(base_dir, path), 'w+') for path in ['train.txt', 'dev.txt', 'test.txt']]
    for question, label in train:
        for char_q, char_l in zip(question, label):
            ftrain.write(f"{char_q}\t{char_l}\n")
        ftrain.write("\n")
    for question, label in dev:
        for char_q, char_l in zip(question, label):
            fdev.write(f"{char_q}\t{char_l}\n")
        fdev.write("\n")
    for question, label in test:
        for char_q, char_l in zip(question, label):
            ftest.write(f"{char_q}\t{char_l}\n")
        ftest.write("\n")
    ftrain.close()
    fdev.close()
    ftest.close()


def write_sim_dataset(train, dev, test, base_dir):
    ftrain, fdev, ftest = [open(osp.join(base_dir, path), 'w+') for path in ['train.txt', 'dev.txt', 'test.txt']]
    for i, item in enumerate(train):
        ftrain.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
    for i, item in enumerate(dev):
        fdev.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
    for i, item in enumerate(test):
        ftest.write("{}\t{}\t{}\t{}\n".format(i, item["question"], item["sim"], item["label"]))
    ftrain.close()
    fdev.close()
    ftest.close()

def main():
    data = generate_data()
    ner_dataset = make_ner_dataset(data)
    sim_dataset = make_sim_dataset(data)
    ner_train_set, ner_dev_set, ner_test_set = split_data(ner_dataset, rseed=20231, ratio=[0.6, 0.2, 0.2])
    sim_train_set, sim_dev_set, sim_test_set = split_data(sim_dataset, rseed=20232, ratio=[0.6, 0.2, 0.2])
    write_ner_dataset(ner_train_set, ner_dev_set, ner_test_set, NER_BASE)
    write_sim_dataset(sim_train_set, sim_dev_set, sim_test_set, SIM_BASE)

if __name__ == '__main__':
    main()