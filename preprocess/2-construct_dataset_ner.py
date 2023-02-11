# coding:utf-8
import sys
import os
import pandas as pd


'''
通过 NLPCC2016KBQA 中的原始数据，构建用来训练NER的样本集合
构造NER训练集，实体序列标注，训练BERT+CRF
'''
data_base = 'models/input/data/fengxian/'
data_dir = data_base + 'qa'
file_name_list = ['train.txt','validate.txt','test.txt']

new_dir = 'ner'

question_str = "<question"
triple_str = "<triple"
answer_str = "<answer"

for file_name in file_name_list:

    q_t_a_list = []
    seq_q_list = []  # ["中","华","人","民"]
    seq_tag_list = []  # [0,0,1,1]

    file_path_name = os.path.join(data_dir,file_name)
    assert os.path.exists(file_path_name)
    with open(file_path_name,'r',encoding='utf-8') as f:
        q_str = ""
        t_str = ""
        a_str = ""
        counter=0
        for line in f:
            if question_str in line:
                q_str = line.strip()
            if triple_str in line:
                t_str = line.strip()
            if answer_str in line:
                a_str = line.strip()
            counter+=1

            # 构建命名实体识别集合 
            # if start_str in line:  # new question answer triple
            # print(t_str)
            if counter % 3==0:
                entities = t_str.split("\t")[0].split(">")[1].strip() # 切分三元组，只有第一个是实体，获取实体
                q_str = q_str.split(">")[1].replace(" ", "").strip()  # 获取问题
                if entities in q_str: 
                    q_list = list(q_str) # 把问题变成列表
                    seq_q_list.extend(q_list) # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
                    seq_q_list.extend([" "]) # 在列表后面添加空格 切分句子 
                    tag_list = ["O" for i in range(len(q_list))]
                    tag_start_index = q_str.find(entities)
                    for i in range(tag_start_index, tag_start_index + len(entities)):
                        if tag_start_index == i:
                            tag_list[i] = "B-entity"
                        else:
                            tag_list[i] = "I-entity"
                    seq_tag_list.extend(tag_list)
                    seq_tag_list.extend([" "])
                else:
                    pass
                q_t_a_list.append([q_str, t_str, a_str]) # 问题 原始三元组 原始答案
                    

    seq_result = [str(q) + " " + tag for q, tag in zip(seq_q_list, seq_tag_list)]
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    with open(os.path.join(new_dir,file_name), "w", encoding='utf-8') as f:
        f.write("\n".join(seq_result))
    f.close()

    df = pd.DataFrame(q_t_a_list, columns=["q_str", "t_str", "a_str"])
    file_type = file_name.split('.')[0]
    csv_name = file_type+'.'+'csv'

    df.to_csv(os.path.join(new_dir,csv_name), encoding='utf-8', index=False)