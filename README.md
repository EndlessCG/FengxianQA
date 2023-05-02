# FengxianQA

风险知识问答系统

## 数据库构建

### Neo4j图数据库构建

知识图谱数据在input下面的风险实体关系表目录中。

其中，entity_fengxiandian.csv、entity_mingci.csv、entity_yewuhuanjie.csv、entity_yewuliucheng.csv、entity_yujing.csv是实体表，relationship_baohan.csv是关系表。

知识图谱数据导入命令（前提：所有数据必须放在neo4j图数据库的import目录下）：

./neo4j-admin import --database=graph.db --nodes="../import/entity_fengxiandian.csv" --nodes="../import/entity_yewuhuanjie.csv" --nodes="../import/entity_yewuliucheng.csv" --nodes="../import/entity_mingci.csv" --nodes="../import/entity_yujing.csv" --relationships="../import/relationship_baohan.csv"


### MySql数据库构建

FAQ数据存放在`input/data/faq/file/qa100.xls`

在config.py的faq_runner_config中配置数据库信息后，运行`python -m preprocess.FAQ.file_to_mysql`即可将数据上传至mysql数据库中

## 使用方法

1. 安装Python和cudnn，cudatoolkit（推荐Python=3.7）

2. 运行 `pip install -r requirements.txt` 安装Python依赖

3. 在 `config.py` 中进行配置，注意在faq_runner_config中设置mysql数据库信息，以及在kbqa_runner_config中设置neo4j数据库信息

4. 运行 `python -m preprocess.FAQ.file_to_mysql`将FAQ数据导入到MySQL数据库中

5. 运行 `bash scripts/train_all.sh` 训练所有模型

   * （可选）运行`bash scripts/test_all.sh`测试模型、模块以及系统的准确率

6. 将FengxianQA添加至PYTHONPATH环境变量中

   * 可以使用`export PYTHONPATH=$PYTHONPATH:<FengxianQA路径>`，但仅在当前命令窗口生效
   * 也可以使用`echo "export PYTHONPATH=$PYTHONPATH:<FengxianQA路径>" >> ~/.bashrc; source ~/.bashrc`，该命令会在每次打开命令窗口后更新PYTHONPATH，从而保证FengxianQA在Python的import查询路径中。

7. 在其它应用中调用FengxianQA： 

   ```Python
   from FengxianQA import FengxianQA
   QA_bot = FengxianQA()
   answer = QA_bot.do_qa("新业务开展前，需要做哪些准备？")
   ```

8. 也可以运行`python fengxian_qa.py`来交互式测试模型
  * 若需要查看系统内部“链接到的实体”、“问题路径”等信息，可在config.py的faq_runner_config, kbqa_runner_config, fengxian_qa_config中将verbose设置为True

## 项目结构

* models
  * EL
    * EL_model.py     实体链接模型
    * EL_main.py      训练实体链接模型
    * test_EL.py      测试实体链接模型
    * EL_ablation_test.py  实体链接模型特征消融测试
  * NER
    * CRF_model.py    条件随机场模型
    * BERT_CRF.py     BERT+条件随机场
    * NER_main.py     训练命令实体识别模型
    * test_NER.py     测试命令实体识别模型
  * SIM
    * SIM_main.py     训练问题-路径相似度模型
    * test_SIM.py     测试问题-路径相似度模型
  * FAQ
    * FAQ_model.py    常见问题匹配模型
    * FAQ_main.py     训练常见问题匹配模型
    * models_util.py  常见问题模型组件库
    * util.py         FAQ模型工具库
* runners
  * bert_kbqa_runner.py KBQA模块定义
  * chatglm_runner.py   ChatGLM模块定义
  * faq_runner.py       FAQ模块定义
  * test_kbqa_runner.py KBQA模块测试脚本
  * test_faq_runner.py  FAQ模块测试脚本
* scripts
  * clean_cached_data.sh 清除训练产生的cached文件模型
  * clean_dev_file.sh    清除开发过程相关文件
  * train_all.sh         训练FAQ, NER以及SIM模型
  * test_all.sh          测试三个模型、两个模块以及系统的性能、系统的响应时间
* input
  * 风控实体关系表        知识图谱原数据
  * data                 训练数据
* utils 工具库
* fengxian_qa.py  FengxianQA类定义
* config.py 项目配置文件
* requirements.txt 项目依赖描述

## 原理简介

* 模型(model)介绍
  * FAQ模型：Bi-LSTM结构的文本分类模型，输入问题原文，输出与问题原文最接近的FAQ数据库中问题的ID
  * NER模型：BERT+CRF结构的命名实体识别模型，输入问题原文，输出问题中每一个字的实体标签
  * EL模型：Xgboost Ranker实体链接模型，输入原问题、实体提及、候选实体，输出候选实体中与实体提及匹配的实体
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
  * ChatGLM模块
    1. 使用ChatGLM生成式语言模型生成回答。
* 输入处理流程：输入的问题首先由FAQ模块进行处理，若FAQ模块除负类外某一类输出的概率大于config.py中指定的`admit_threshold`，则返回FAQ模块的答案，否则将问题输入KBQA模块。若KBQA模块中的各步骤均正常输出，则返回KBQA的答案，否则返回ChatGLM生成的答案。