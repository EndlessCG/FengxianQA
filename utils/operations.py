import argparse
from copy import deepcopy

def load_sim_questions(sim_path):
    questions = set()
    with open(sim_path, 'r') as f:
        for line in f.readlines():
            questions.add(line.split('\t')[1])
    return list(questions)

def load_faq_questions(faq_path):
    questions = set()
    with open(faq_path, 'r') as f:
        for line in f.readlines():
            questions.add(''.join(line.split('\t')[-1].split(' ')).split('\n')[0])
    return list(questions)

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

