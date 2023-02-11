# bert-kbqa
基于bert的kbqa系统


构造数据集

> 数据预处理脚本需要在项目根目录下执行

通过 1_split_data.py 切分数据

通过 2-construct_dataset_ner.py 构造命名实体识别的数据集

通过 3-construct_dataset_attribute.py 构造属性相似度的数据集

通过 4-print-seq-len.py 看看句子的长度



CRF_Model.py  条件随机场模型

BERT_CRF.py  bert+条件随机场

NER_main.py  训练命令实体识别的模型

SIM_main.py  训练属性相似度的模型


test_NER.py  测试命令实体识别

test_SIM.py 测试属性相似度

fengxian_main.py  测试整个项目


主要依赖版本：

torch.__version__    1.2.0

transformers.__version__   2.0.0





