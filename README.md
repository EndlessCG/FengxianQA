# FengxianQA
风险知识问答系统
## 使用方法
1. 安装Python和cudnn，cudatoolkit（推荐Python=3.7 cudnn=7.3.1-cuda10.0_0 cudatoolkit=10.0.130）
    * 如使用Anaconda，可执行如下命令安装cudnn和cudatoolkit：
    ```bash
    $ wget https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/linux-64/cudatoolkit-10.0.130-0.conda
    wget 
    $ conda install cudatoolkit-10.0.130-0.conda
    $ wget https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/linux-64/cudnn-7.3.1-cuda10.0_0.conda
    $ conda install cudatoolkit-7.3.1-cuda10.0_0.conda
    $ rm cudatoolkit-10.0.130-0.conda
    $ rm cudatoolkit-7.3.1-cuda10.0_0.conda
    ```
2. 运行 `pip install -r requirements.txt` 安装Python依赖
3. 运行 `bash scripts/trian_all.sh` 训练所有模型
4. 在 `config.py` 中进行配置，注意在faq_runner_config中设置mysql数据库信息，以及在kbqa_runner_config中设置neo4j数据库信息
5. 将FengxianQA添加至PYTHONPATH环境变量中
    * 可以使用`export PYTHONPATH=$PYTHONPATH:<FengxianQA路径>`，但仅在当前命令窗口生效
    * 也可以使用`echo "export PYTHONPATH=$PYTHONPATH:<FengxianQA路径>" >> ~/.bashrc; source ~/.bashrc`
6. 在其它应用中调用FengxianQA： 
    ```Python
    from FengxianQA import FengxianQA
    QA_bot = FengxianQA()
    answer = QA_bot.do_qa("新业务开展前，需要做哪些准备？")
    ```

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

## 原理简介
* 模型(model)介绍
    * FAQ模型：BiDirectional-LSTM结构的文本分类模型，输入问题原文，输出与问题原文最接近的FAQ数据库中问题的ID
    * NER模型：BERT+CRF结构的命名实体识别模型，输入问题原文，输出问题中每一个字的实体标签
    * SIM模型：基于BERT的文本分类模型，输入问题原文和子图，输出二者是否匹配
* 模块(runner)介绍
    * FAQ（Frequently Asked Questions，常用问题库）模块
        1. 使用FAQ模型获取与输入问题最匹配的FAQ ID
        2. 根据1得到的ID从数据库中获取答案，返回答案以及概率
    * KBQA（Knowledge Base Question-Answering，知识图谱问答）模块
        1. 使用NER模型进行命名实体识别。
        2. 将1中得到的实体与neo4j数据库中存在的实体进行实体链接。
        3. 根据链接到的实体生成候选查询路径子图。
        4. 使用SIM模型判定所有候选字图与问题的匹配度。
        5. 使用最匹配的子图结构生成查询语句。
        6. 在neo4j数据库中进行查询。
        7. 根据查询结果生成答案。
* 输入处理流程：输入的问题首先由FAQ模块进行处理，若FAQ模块输出的概率大于config.py中指定的`admit_threshold`，则返回FAQ模块的答案，否则将问题输入KBQA模块，并返回KBQA模块的答案。