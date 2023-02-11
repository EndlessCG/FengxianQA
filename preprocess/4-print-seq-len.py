import os

# 计算模型最大的长度多少合适

data_base = 'models/input/data/fengxian/'
dir_name = data_base + 'sim_data'
file_list = ['train.txt','validate.txt','test.txt']

for file in file_list:

    file_path_name = os.path.join(dir_name,file)

    max_len = 0
    print("****** {} *******".format(file))
    with open(file_path_name,'r',encoding='utf-8') as f:
        for line in f:

            line_list = line.split('\t')
            question = list(line_list[1])
            attribute = list(line_list[2])
            add_len = len(question) + len(attribute)
            if add_len > max_len:
                max_len = add_len
    print("max_len",max_len)
    f.close()

