import argparse
import os
from copy import deepcopy, copy

KBQA_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def get_abs_path(rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    else:
        return os.path.join(KBQA_BASE_DIR, rel_path)

def convert_config_paths(config: dict):
    for k, v in config.items():
        if isinstance(v, dict):
            convert_config_paths(v)
        elif "file" in k or "path" in k or "dir" in k:
            if v != "":
                config[k] = get_abs_path(v)

def get_all_sims(sim_path):
    sims = set()
    with open(sim_path, 'r') as f:
        for line in f.readlines():
            sims.add(line.split('\t')[2])
    return list(sims)

def load_sim_questions(sim_path, get_answers=False):
    questions = []
    with open(sim_path, 'r') as f:
        for line in f.readlines():
            question = line.split('\t')[1]
            sgraph = line.split('\t')[-2]
            match = line.split('\t')[-1]
            if match == '1\n' and question not in questions:
                questions.append([question, sgraph] if get_answers else question)
    return questions

def load_ner_questions(ner_path, get_answers=False):
    questions = []
    with open(ner_path, 'r') as f:
        question = ""
        label = []
        for line in f.readlines():
            line_list = line.split(' ')
            if len(line_list) == 2:
                question += line_list[0]
                label.append(line_list[1][:-1])
            elif len(question) != 0 and len(label) != 0:
                questions.append([question, label] if get_answers else question)
                question = ""
                label = []
    return questions

def load_faq_questions(faq_path, get_answers=False):
    questions = []
    with open(faq_path, 'r') as f:
        for line in f.readlines():
            question = ''.join(line.split('\t')[-1].split(' ')).split('\n')[0]
            answer = line.split('\t')[0]
            if question not in questions and answer != "-1":
                if get_answers:
                    questions.append([question, answer])
                else:
                    questions.append(question)
    return questions

def load_el_questions(el_path):
    questions = {}
    with open(el_path, 'r') as f:
        for line in f.readlines():
            if len(line.split('\t')) != 5:
                print(f"invalid el data {line}")
                continue
            mention, entity, question, label, original = line.split('\t')
            if label == '1':
                original = original.replace('\n', '')
                questions.setdefault(original, []).append([mention, question, entity])
    return questions

def merge_arg_and_config(merge1, merge2):
    if isinstance(merge1, argparse.Namespace):
        merge1_c = vars(merge1)
    else:
        merge1_c = deepcopy(merge1)
    
    if isinstance(merge2, argparse.Namespace):
        merge2_c = vars(merge2)
    else:
        merge2_c = deepcopy(merge2)
    
    # merge2 has higher priority
    merge1_c.update(merge2_c)
    return argparse.Namespace(**merge1_c)

