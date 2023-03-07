from fengxian_qa import FengxianQA
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from matplotlib import rcParams

from runners.bert_kbqa_runner import BertKBQARunner
from runners.faq_runner import FAQRunner
from config import *
from .operations import load_faq_questions, load_sim_questions

rcParams['font.family'] = 'SimHei'

time_profile_ninputs = [10, 50, 100, 200]

def plot_avg_time(times, topic):
    plt.hist(times)
    plt.show()
    plt.xlabel("请求次数")
    plt.ylabel("平均请求时间(s)")
    plt.savefig(topic)

def faq_once():
    faq_questions = load_faq_questions("input/data/fengxian/faq/train_data")
    agent = FengxianQA()
    agent.do_qa(faq_questions[0])

def kbqa_once():
    questions = load_sim_questions("input/data/fengxian/sim/train.txt")
    agent = FengxianQA()
    agent.do_qa(questions[0])

def test_init_time(ntrials):
    kbqa_init_time, faq_init_time, faq_model_init_time, ner_init_time, sim_init_time = [], [], [], [], []
    fengxianqa_init_time = []
    for _ in tqdm(range(ntrials)):
        start = time.time()
        agent = FengxianQA()
        end = time.time()
        fengxianqa_init_time.append(end - start)
        del agent
    print("fengxianqa", fengxianqa_init_time)
    print(np.average(fengxianqa_init_time), "+-", np.std(fengxianqa_init_time))
    for _ in tqdm(range(ntrials)):
        start = time.time()
        kbqa_runner = BertKBQARunner(kbqa_runner_config)
        end = time.time()
        kbqa_init_time.append(end - start)
        del kbqa_runner
    print("kbqa runner", kbqa_init_time)
    print(np.average(kbqa_init_time), "+-", np.std(kbqa_init_time))
    for _ in tqdm(range(ntrials)):
        start = time.time()
        faq_runner = FAQRunner(faq_runner_config)
        end = time.time()
        faq_init_time.append(end - start)
        del faq_runner
    print("faq runner", faq_init_time)
    print(np.average(faq_init_time), "+-", np.std(faq_init_time))
    for _ in tqdm(range(ntrials)):
        faq_runner = FAQRunner(faq_runner_config)
        start = time.time()
        faq_runner._load_faq_model()
        end = time.time()
        faq_model_init_time.append(end - start)
        del faq_runner
    print("faq model", faq_model_init_time)
    print(np.average(faq_model_init_time), "+-", np.std(faq_model_init_time))
    for _ in tqdm(range(ntrials)):
        kbqa_runner = BertKBQARunner(kbqa_runner_config)
        start = time.time()
        kbqa_runner._load_ner_model(kbqa_runner_config["ner"])
        end = time.time()
        ner_init_time.append(end - start)
        del kbqa_runner
    print("ner model", ner_init_time)
    print(np.average(ner_init_time), "+-", np.std(ner_init_time))
    for _ in tqdm(range(ntrials)):
        kbqa_runner = BertKBQARunner(kbqa_runner_config)
        start = time.time()
        kbqa_runner._load_sim_model(kbqa_runner_config["sim"])
        end = time.time()
        sim_init_time.append(end - start)
        del kbqa_runner
    print("sim model", sim_init_time)
    print(np.average(sim_init_time), "+-", np.std(sim_init_time))
    


def main():
    # test_init_time(ntrials=10)
    agent = FengxianQA()
    questions = load_sim_questions("input/data/fengxian/sim/train.txt")
    faq_questions = load_faq_questions("input/data/fengxian/faq/train_data")
    for ninput in time_profile_ninputs:
        times = []
        print("ninput:", ninput)
        for q in tqdm(questions[:ninput]):
            t_start = time.time()
            _ = agent.do_qa(q)
            t_end = time.time()
            times.append(t_end - t_start)
        print("total time", np.sum(times[1:]))
        print("full", np.average(times[1:]), "+-", np.std(times[1:]))
    for ninput in time_profile_ninputs:
        times = []
        print("ninput:", ninput)
        for q in tqdm(questions[:ninput]):
            t_start = time.time()
            _ = agent.do_qa_without_faq(q)
            t_end = time.time()
            times.append(t_end - t_start)
        print("total time", np.sum(times[1:]))
        print("kbqa", np.average(times[1:]), "+-", np.std(times[1:]))
    # plot_avg_time(times, "./kbqa_avgtime.png")
    for ninput in time_profile_ninputs:
        times = []
        print("ninput:", ninput)
        for q in tqdm(questions[:ninput]):
            t_start = time.time()
            _ = agent.kbqa_runner.get_entity(q)
            t_end = time.time()
            times.append(t_end - t_start)
        print("total time", np.sum(times[1:]))
        print("ner", np.average(times[1:]), "+-", np.std(times[1:]))
    for ninput in time_profile_ninputs:
        times = []
        print("ninput:", ninput)
        for q in tqdm(faq_questions[:ninput]):
            t_start = time.time()
            _ = agent.do_qa(q)
            t_end = time.time()
            times.append(t_end - t_start)
        print("total time", np.sum(times[1:]))
        print("faq", np.average(times[1:]), "+-", np.std(times[1:]))
    # plot_avg_time(times, "./faq_avgtime.png")

if __name__ == '__main__':
    if '--faq_once' in sys.argv:
        faq_once()
    elif '--kbqa_once' in sys.argv:
        kbqa_once()
    else:
        main()
