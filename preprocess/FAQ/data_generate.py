import os
import pandas as pd
import random
import jieba
from tqdm import tqdm

from preprocess.utils import get_synonyms
from utils.operations import load_sim_questions
from config import sim_model_config 
DATA_BASE = "input/data/faq/no_commas_large_neg2pos_1/"

# 同义词替换(Synonyms Replace)
def sr(num, dataset):
    import synonyms
    times = 0
    data_n = []
    total_same = 0
    same_gate = 5
    total_before = -1  # 记录上一次分类的个数
    while times < num:
        # 单个词的替换
        for index, data in enumerate(dataset):
            # 分词
            div_list = jieba.lcut(data)
            up = len(div_list) - 1
            sr_pos = random.randint(0, up)
            sr_words = synonyms.nearby(div_list[sr_pos])
            # 如果没有结果，直接跳过
            if len(sr_words[0]) == 0:
                continue
            words_list = sr_words[0]
            p_list = sr_words[1]
            for index_p, p in enumerate(p_list):
                if 1.0 > p > 0.3:
                    new_data = div_list.copy()
                    new_data[sr_pos] = words_list[index_p]
                    new_data = ''.join(i for i in new_data)
                    data_n.append(new_data)
                    data_n = list(set(data_n))  # 去重
                    times = len(data_n)
                    if times >= num:
                        break
                if times >= num:
                    break
        if times >= num:
            break
        # 双个单词替换：如果上一轮不等于当前的，不管，更新上一轮个数后继续
        if total_before != times:
            total_before = times
        # 如果等于上一轮，则进行两个位置的单词替换
        for index, data in enumerate(dataset):
            # 分词
            div_list = jieba.lcut(data)
            mid = int(len(div_list) / 2)
            # 获取两个区间
            fir_pos, sec_pos = random.randint(0, mid - 1), random.randint(mid, len(div_list) - 1)
            # 获取两个位置的近义词词组
            fir_words, sec_words = synonyms.nearby(div_list[fir_pos]), synonyms.nearby(div_list[sec_pos])
            if len(fir_words) == 0 or len(sec_words) == 0:
                break
            fir_p_list, fir_words_list = fir_words[1], fir_words[0]
            sec_p_list, sec_words_list = sec_words[1], sec_words[0]
            for index_fir_p, fir_p in enumerate(fir_p_list):
                for index_sec_p, sec_p in enumerate(sec_p_list):
                    new_data = div_list.copy()
                    # 替换第一个词
                    if 1.0 > fir_p > 0.3:
                        new_data[fir_pos] = fir_words_list[index_fir_p]
                    # 替换第二个词
                    if 1.0 > sec_p > 0.3:
                        new_data[sec_pos] = sec_words_list[index_sec_p]
                    new_data = ''.join(i for i in new_data)
                    data_n.append(new_data)
                    data_n = list(set(data_n))
                    times = len(data_n)
                    if total_before == times:
                        total_same += 1
                    else:
                        total_before = times
                    if times >= num:
                        break
                if times >= num:
                    break
            if times >= num:
                break
        if total_same >= same_gate:
            print('sr过程因无法扩充更多数据中途停止，最终的扩充条数为：', total_before, '条')
            break
        if times >= num:
            break
    return [i.replace("\t", "") for i in data_n]


# 随机插入-随机抽一个词，在该词的同义词集合中选一个，随机插入到原句子中的随机位置
def ri(num, dataset):
    import synonyms
    data_n = []
    times = 0
    times_before = -1
    while times < num:
        for index, data in enumerate(dataset):
            # 分词
            div_list = jieba.lcut(data)
            up = len(div_list) - 1
            ri_pos = random.randint(0, up)  # 获取要抽取的相似度词
            ri_words = synonyms.nearby(div_list[ri_pos])
            # 如果没有结果，直接跳过
            if len(ri_words[0]) == 0:
                continue
            words_list = ri_words[0]  # 同义词表
            p_list = ri_words[1]  # 同义词对应相似度
            for index_p, p in enumerate(p_list):
                if 1.0 > p > 0.5:
                    insert_pos = random.randint(0, up)  # 获取词要插入的位置
                    new_data = div_list.copy()
                    new_data.insert(insert_pos, words_list[index_p])
                    new_data = ''.join(i for i in new_data)
                    data_n.append(new_data)
                    data_n = list(set(data_n))  # 去重
                    times = len(data_n)
                    if times >= num:
                        break
            if times_before == times:
                break
            else:
                times_before = times
            if times >= num:
                break
        if times_before == times:
            print('ri过程因无法扩充更多数据中途停止，最终的扩充条数为：', times_before, '条')
            break
        if times >= num:
            break
    return [i.replace("\t", "") for i in data_n]


# rs过程 - 随机选择两个词，位置交换
def rs(num, dataset):
    data_n = []
    times = 0
    total_same = 0  # 记录上一次分类的个数
    same_gate = 5  # 如果5次相同，则证明没有扩充的余地了
    total_before = -1
    while times < num:
        # 单个单词替换
        for index, data in enumerate(dataset):
            # 分词
            div_list = jieba.lcut(data)
            mid = int((len(div_list)) / 2)
            first_pos = random.randint(0, mid)
            second_pos = random.randint(mid, len(div_list) - 1)
            div_list[first_pos], div_list[second_pos] = div_list[second_pos], div_list[first_pos]
            new_data = ''.join([i for i in div_list])
            data_n.append(new_data)
            data_n = list(set(data_n))  # 去重
            times = len(data_n)
            if times >= num:
                break
        # 双个单词替换：如果上一轮不等于当前的，不管，更新上一轮个数后继续
        if total_before != times:
            total_before = times
        # 如果等于上一轮，则进行两个位置的单词替换
        for index, data in enumerate(dataset):
            div_list = jieba.lcut(data)
            break_forloop = False
            if len(div_list) < 4:
                break_forloop = True
                break
            mid = int((len(div_list)) / 2)
            fir_mid, second_mid = int((0 + mid) / 2), int((mid + len(div_list)) / 2)
            fir_pos, sec_pos, third_pos, forth_pos = random.randint(0, fir_mid - 1), random.randint(fir_mid,
                                                                                                    mid - 1), random.randint(
                mid, second_mid - 1), random.randint(second_mid, len(div_list) - 1)
            # 交换 1-4\2-3
            div_list[fir_pos], div_list[forth_pos] = div_list[forth_pos], div_list[fir_pos]
            div_list[sec_pos], div_list[third_pos] = div_list[third_pos], div_list[sec_pos]
            # 第二种交换思路1-3\2-4
            new_data_2 = div_list.copy()
            new_data_2[fir_pos], new_data_2[third_pos] = new_data_2[third_pos], new_data_2[fir_pos]
            new_data_2[sec_pos], new_data_2[forth_pos] = new_data_2[forth_pos], new_data_2[sec_pos]
            # 合并为字符串
            new_data = ''.join([i for i in div_list])
            new_data_2 = ''.join([i for i in new_data_2])
            # 添加
            data_n.append(new_data)
            data_n.append(new_data_2)
            data_n = list(set(data_n))  # 去重
            times = len(data_n)  # 计数
            if times >= num:
                break
            # 如果更改还是没有改变则退出大循环
            if total_before == times:
                total_same += 1
                break
            else:
                total_before = times
        if total_same >= same_gate or break_forloop:
            print('rs过程因无法扩充更多数据中途停止，最终的扩充条数为：', times, '条')
            break
        if times >= num:
            break
        print('times=', times)
    return [i.replace("\t", "") for i in data_n]


# 生成训练集、验证集、测试集（利用数据增强），包含标准问题ID、扩展问题ID、扩展问题文本三列
def generate_datasets(strategy='baidufanyi'):
    assert strategy in ['openai', 'baidufanyi'], 'Invalid strategy'
    to_write_train = ""
    to_write_dev = ""
    to_write_test = ""
    train_list = []
    dev_list = []
    test_list = []
    train_split = 0.8
    dev_split = 0.1
    train_number = []
    dev_number = []
    test_number = []
    from back_translation import back_trans
    with open(f'{DATA_BASE}/std_data', 'r') as f:
        data = f.readlines()
        for i in tqdm(data):
            l = i.split('\t')
            biao_id = l[1]
            to_text = " ".join(l[2:])
            text = to_text.replace("\n", "").replace(" ", "")

            if strategy == 'baidufanyi':
                import synonyms
                generate_list = sr(5, [text])
                generate_list.extend(ri(5, [text]))
                generate_list.extend(rs(5, [text]))
                generate_list.extend([back_trans(text, c) for c in ['en','jp','kor','spa','fra']])

            elif strategy == 'openai':
                generate_list = []
                synonyms = get_synonyms(text.replace("，", ""), 50)
                if synonyms is not None:
                    generate_list.extend(synonyms)
            
            generate_list = [" ".join(list(q)) for q in generate_list if len(q) >= 3]
            fir_split = int(train_split * len(generate_list))
            sec_split = fir_split + int(dev_split * len(generate_list))
            train_list.extend(generate_list[:fir_split])
            dev_list.extend(generate_list[fir_split:sec_split])
            test_list.extend(generate_list[sec_split:])
            train_number.extend([biao_id] * len(generate_list[:fir_split]))
            dev_number.extend([biao_id] * len(generate_list[fir_split:sec_split]))
            test_number.extend([biao_id] * len(generate_list[sec_split:]))

    with open(f'{DATA_BASE}/train_data', 'w') as f:
        kuo_id = 0
        for j in range(len(train_list)):
            to_write_train += str(train_number[j]) + "\t" + str(kuo_id) + "\t" + train_list[j].replace("\n", "") + "\n"
            kuo_id += 1
        f.write(to_write_train)
    with open(f'{DATA_BASE}/dev_data', 'w') as f:
        kuo_id = 0
        for j in range(len(dev_list)):
            to_write_dev += str(dev_number[j]) + "\t" + str(kuo_id) + "\t" + dev_list[j].replace("\n", "") + "\n"
            kuo_id += 1
        f.write(to_write_dev)
    with open(f'{DATA_BASE}/test_data', 'w') as f:
        kuo_id = 0
        for j in range(len(test_list)):
            to_write_test += str(test_number[j]) + "\t" + str(kuo_id) + "\t" + test_list[j].replace("\n", "") + "\n"
            kuo_id += 1
        f.write(to_write_test)

def expand_neg_questions(file_name, sim_file_name):
    to_write_train = ""
    to_write_dev = ""
    to_write_test = ""
    train_list = []
    dev_list = []
    test_list = []
    train_split = 0.8
    dev_split = 0.1
    train_number = []
    dev_number = []
    test_number = []
    generate_dict = {}

    if isinstance(file_name, str):
        file_name = [file_name]
    if isinstance(sim_file_name, str):
        sim_file_name = [sim_file_name]
    
    for file in file_name:
        with open(file, "r") as f:
            for line in f.readlines():
                question = line.split('\t')[-1].replace(" ", "").replace("\n", "")
                id_ = line.split('\t')[0]
                generate_dict.setdefault(id_, []).append(' '.join(question))
    
    for file in sim_file_name:
        questions = [' '.join(q) for q in load_sim_questions(file)]
        generate_dict.setdefault(-1, []).extend(questions)

    for id_, questions in generate_dict.items():
        fir_split = int(train_split * len(questions))
        sec_split = fir_split + int(dev_split * len(questions))
        train_list.extend(questions[:fir_split])
        dev_list.extend(questions[fir_split:sec_split])
        test_list.extend(questions[sec_split:])
        train_number.extend([id_] * len(questions[:fir_split]))
        dev_number.extend([id_] * len(questions[fir_split:sec_split]))
        test_number.extend([id_] * len(questions[sec_split:]))

    with open(f'{DATA_BASE}/train_data', 'w') as f:
        kuo_id = 0
        for j in range(len(train_list)):
            to_write_train += str(train_number[j]) + "\t" + str(kuo_id) + "\t" + train_list[j].replace("\n", "") + "\n"
            kuo_id += 1
        f.write(to_write_train)
    with open(f'{DATA_BASE}/dev_data', 'w') as f:
        kuo_id = 0
        for j in range(len(dev_list)):
            to_write_dev += str(dev_number[j]) + "\t" + str(kuo_id) + "\t" + dev_list[j].replace("\n", "") + "\n"
            kuo_id += 1
        f.write(to_write_dev)
    with open(f'{DATA_BASE}/test_data', 'w') as f:
        kuo_id = 0
        for j in range(len(test_list)):
            to_write_test += str(test_number[j]) + "\t" + str(kuo_id) + "\t" + test_list[j].replace("\n", "") + "\n"
            kuo_id += 1
        f.write(to_write_test)

if __name__ == "__main__":
    expand_neg_questions(["input/data/faq/no_commas_large/train_data",
    "input/data/faq/no_commas_large/dev_data",
    "input/data/faq/no_commas_large/test_data"],
    ["input/data/sim/train.txt",
    "input/data/sim/validate.txt",
    "input/data/sim/test.txt"])

