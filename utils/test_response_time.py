from fengxian_qa import FengxianQA
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

time_profile_ninputs = [500]
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

def plot_avg_time(times, topic):
    plt.hist(times)
    plt.show()
    plt.xlabel("请求次数")
    plt.ylabel("平均请求时间(s)")
    plt.savefig(topic)

<<<<<<< Updated upstream
=======
def faq_once():
    faq_questions = load_faq_questions("input/data/fengxian/faq/train_data")
    agent = FengxianQA()
    agent.do_qa(faq_questions[0])

def kbqa_once():
    questions = load_sim_questions("input/data/fengxian/sim/train.txt")
    agent = FengxianQA()
    agent.do_qa(questions[0])

>>>>>>> Stashed changes
def main():
    agent = FengxianQA()
    questions = load_sim_questions("input/data/fengxian/sim/train.txt")
    faq_questions = load_faq_questions("input/data/fengxian/faq/train_data")
    times = []
    for ninput in time_profile_ninputs:
        for q in tqdm(questions[:ninput]):
            t_start = time.time()
            _ = agent.do_qa(q)
<<<<<<< Updated upstream
        t_end = time.time()
        times[ninput] = (t_end - t_start) / ninput
    plot_avg_time(times, "./kbqa_avgtime.png")
    times = {}
=======
            t_end = time.time()
            times.append(t_end - t_start)
    print("kbqa", times)
    # plot_avg_time(times, "./kbqa_avgtime.png")
    times = []
>>>>>>> Stashed changes
    for ninput in time_profile_ninputs:
        for q in tqdm(faq_questions[:ninput]):
            t_start = time.time()
            _ = agent.do_qa(q)
<<<<<<< Updated upstream
        t_end = time.time()
        times[ninput] = (t_end - t_start) / ninput
    plot_avg_time(times, "./faq_avgtime.png")

if __name__ == '__main__':
    main()
=======
            t_end = time.time()
            times.append(t_end - t_start)
    print("faq", times)
    # plot_avg_time(times, "./faq_avgtime.png")

if __name__ == '__main__':
    if '--faq_once' in sys.argv:
        faq_once()
    elif '--kbqa_once' in sys.argv:
        kbqa_once()
    else:
        main()
>>>>>>> Stashed changes
