from fengxian_qa import FengxianQA
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

time_profile_ninputs = [1, 5, 10, 20, 50, 100, 200, 300, 500]
def load_sim_questions(sim_path):
    questions = set()
    with open(sim_path, 'r') as f:
        for line in f.readlines():
            questions.add(line.split('\t')[-1])
    return list(questions)

def load_faq_questions(faq_path):
    questions = set()
    with open(faq_path, 'r') as f:
        for line in f.readlines():
            questions.add(''.join(line.split('\t')[-1].split(' ')).split('\n')[0])
    return list(questions)

def plot_avg_time(times, topic):
    plt.hist(times)
    plt.show()
    plt.xlabel("请求次数")
    plt.ylabel("平均请求时间(s)")
    plt.savefig(topic)

def main():
    agent = FengxianQA()
    questions = load_sim_questions("input/data/fengxian/sim/train.txt")
    faq_questions = load_faq_questions("input/data/fengxian/faq/train_data")
    times = {}
    for ninput in time_profile_ninputs:
        t_start = time.time()
        for q in questions[:ninput]:
            _ = agent.do_qa(q)
        t_end = time.time()
        times[ninput] = (t_end - t_start) / ninput
    plot_avg_time(times, "./kbqa_avgtime.png")
    times = {}
    for ninput in time_profile_ninputs:
        t_start = time.time()
        for q in faq_questions[:ninput]:
            _ = agent.do_qa(q)
        t_end = time.time()
        times[ninput] = (t_end - t_start) / ninput
    plot_avg_time(times, "./faq_avgtime.png")

if __name__ == '__main__':
    main()