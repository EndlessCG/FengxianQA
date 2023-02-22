记得先cd到FAQ路径下
# 数据准备[已完成]
运行data/process文件夹下的process_std.py, 生成 std_data 和 ans_data, 再运行 data/process文件夹下的 data_generate.py, 产生 train_data, dev_data 和 test_data
# 模型预训练和微调
## 模型预训练
运行1_run_pretraining.py, 预训练语料和词典在data文件夹下
## 模型微调
运行2_run_classifier.py
# 模型评估,预测和使用
## 模型评估
运行3_run_eval.py, 输出准确率
## 模型预测单条样本
运行4_run_predict.py, 输出对应回答