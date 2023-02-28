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
4. 在 `config.py` 中进行配置
5. 运行 `python fengxian_qa.py` 测试问答模型效果
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
* TODO 待完善