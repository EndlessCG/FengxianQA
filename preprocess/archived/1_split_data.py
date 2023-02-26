import numpy as np
import random

def getRandomIndex(n, x):
	# 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index

data_base = 'input/data/fengxian/'
data_sets = [data_base + p for p in ["qa/QA_unchained_mhop_data.txt", "qa/QA_mhop_data.txt", "qa/QA_data.txt"]]
train_set = data_base + "qa/train.txt"
validate_set = data_base + "qa/validate.txt"
test_set = data_base + "qa/test.txt"
temp_set = data_base + "qa/temp.txt"

write_train =open(train_set,"w")
write_test =open(test_set,"w")
write_validate= open(validate_set,"w")
write_temp= open(temp_set,"w")

datasets = []
for data_set in data_sets:
    with open(data_set, "r") as f:
        datasets.extend(f.readlines())

datas_num = len(datasets)
# 先根据上面的函数获取train_index
train_index = np.array(getRandomIndex(datas_num, int(datas_num * 0.6))) # 6:2:2  
# print(len(train_index))

# 再讲train_index从总的index中减去就得到了test_index
test_index = np.delete(np.arange(datas_num), train_index)
# print(len(train_index))

counter =0
i =0 
for data in datasets:
    if "<question " in data:
        counter+=1
    if counter in train_index:
        write_train.write(data)
    else:
        write_temp.write(data)
    i +=1

f.close()
write_train.close()
write_temp.close()

f_temp = open(temp_set, "r")
dataset= f_temp.readlines()

counter =0
i =0 
for data in dataset:
    if "<question " in data:
        counter+=1
    if counter % 2 ==0:
        write_validate.write(data)
    else:
        write_test.write(data)
    i +=1

f_temp.close()
write_test.close()
write_validate.close()