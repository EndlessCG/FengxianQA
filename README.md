# bert-kbqa
基于bert的kbqa系统
* Python=3.7 cudnn=7.3.1-cuda10.0_0 cudatoolkit=10.0.130
## 项目结构

* models
    * NER
        * CRF_model.py  条件随机场模型
        * BERT_CRF.py  bert+条件随机场
        * NER_main.py  训练命令实体识别的模型
        * test_NER.py  测试命令实体识别
    * SIM
        * SIM_main.py  训练属性相似度的模型
        * test_SIM.py 测试属性相似度
    * FAQ
        * FAQ_model.py
        * FAQ_main.py
        * models_util.py
        * util.py
* runners
    * bert_kbqa_runner.py
    * faq_runner.py
* scripts
    * clean_cached_data.sh
    * train_all.sh
* preprocess 用于生成训练数据，训练完成后部署时可删除
* utils 工具包
* fengxian_main.py  FengxianQA类定义
* config.py 项目配置文件
* requirements.txt 项目依赖描述

## 使用方法
* 训练所有模型
    1. 运行 `pip install -r requirements.txt`安装依赖
    2. 

