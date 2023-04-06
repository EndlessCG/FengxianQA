# FengxianQA项目配置文件
# 若部分参数未在本文件中定义，将采用对应模块原文件中给出的默认值

neo4j_config=dict(
    neo4j_addr="bolt://localhost:7687",  # neo4j地址
    username="neo4j",  # neo4j用户名
    password="123456",  # neo4j密码
)

# 以下为模型配置，仅在模型训练与测试时有效，应在模型训练前配置
ner_model_config = dict(
    # NER模型训练与测试配置
    # 此处配置仅影响NER模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    train=dict(
        # 更多参数定义请参考models/NER/NER_main.py或运行python -m models.NER.NER_main --help
        data_dir="input/data/ner",  # 实体识别输入数据目录
        pre_train_model_file="input/pretrained_BERT/bert-base-chinese-model.bin",  # BERT预训练模型路径
        vob_file="input/pretrained_BERT/bert-base-chinese-vocab.txt",  # BERT预训练模型词汇表路径
        model_config_file="input/pretrained_BERT/bert-base-chinese-config.json",  # BERT预训练模型配置文件路径
        output_dir="models/NER/ner_output",  # 训练好的实体识别模型保存路径
        max_seq_length=128,  # 最大序列长度
        train_batch_size=32,  # 训练时batch size
        eval_batch_size=32,  # 训练时测试batch size
        gradient_accumulation_steps=4,  # 梯度累加步数
        learning_rate=5e-5,  # 学习率
        weight_decay=0.0,  # 权重衰减系数
        adam_epsilon=1e-8,  # Adam优化器epsilon
        max_grad_norm=1.0,  # 最大梯度更新
        seed=42,  # 初始化用随机种子
        warmup_steps=0,  # Adam优化器warmup步数
        num_train_epochs=5,  # 训练迭代数
    ),
    test=dict(
        model_path='models/NER/ner_output/best_ner.bin',  # 训练好的实体识别模型保存路径
        data_dir="input/data/ner/",  # 训练过程中生成的cached测试数据所在文件夹
        raw_data_path="input/data/ner/test.txt",  # 测试数据原文件路径
        vob_file="input/pretrained_BERT/bert-base-chinese-vocab.txt",  # BERT预训练模型词汇表路径
        model_config_file="input/pretrained_BERT/bert-base-chinese-config.json",  # BERT预训练模型配置文件路径
        max_seq_length=128,  # 最大序列长度
    )
)

el_model_config = dict(
    train=dict(
        train_file="input/data/el/train.txt",
        dev_file="input/data/el/validate.txt",
        output_dir="models/EL/el_output/",
        w2v_corpus_path="input/data/el/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt",
        w2v_load_path="models/EL/el_output/w2v_model.bin",
        w2v_save_path="models/EL/el_output/w2v_model.bin",
    ),
    test=dict(
        test_file="input/data/el/test.txt",
        w2v_load_path="models/EL/el_output/w2v_model.bin",
        model_path="models/EL/el_output/best_el.bin"
    )
)

sim_model_config = dict(
    # SIM模型训练配置
    # 此处配置仅影响SIM模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    train=dict(
        # 更多参数定义请参考models/SIM/SIM_main.py或运行python -m models.SIM.SIM_main --help
        data_dir="input/data/sim",  # 输入数据目录
        pre_train_model_file="input/pretrained_BERT/bert-base-chinese-model.bin",  # BERT预训练模型路径
        vob_file="input/pretrained_BERT/bert-base-chinese-vocab.txt",  # BERT预训练模型词汇表路径
        model_config_file="input/pretrained_BERT/bert-base-chinese-config.json",  # BERT预训练模型配置文件路径
        output_dir="models/SIM/sim_output/",  # 训练好的句子分类模型保存路径
        max_seq_length=128,  # 最大序列长度
        train_batch_size=32,  # 训练时batch size
        eval_batch_size=32,  # 训练时测试batch size
        gradient_accumulation_steps=4,  # 梯度累加步数
        learning_rate=5e-5,  # 学习率
        weight_decay=0.0,  # 权重衰减系数
        adam_epsilon=1e-8,  # Adam优化器epsilon
        max_grad_norm=1.0,  # 最大梯度更新
        seed=42,  # 初始化用随机种子
        warmup_steps=0,  # Adam优化器warmup步数
        num_train_epochs=5,  # 训练迭代数
    ),
    test=dict(
        # 更多参数定义请参考models/SIM/test_SIM.py或运行python -m models.SIM.test_SIM --help
        model_path='models/SIM/sim_output/best_sim.bin',  # 训练好的句子分类模型保存路径
        data_dir="input/data/sim/",  # 训练过程中生成的cached测试数据所在文件夹
        vob_file="input/pretrained_BERT/bert-base-chinese-vocab.txt",  # BERT预训练模型词汇表路径
        model_config_file="input/pretrained_BERT/bert-base-chinese-config.json",  # BERT预训练模型配置文件路径
        max_seq_length=128,  # 最大序列长度
    )
)

faq_model_config = dict(
    # FAQ模型训练配置
    # 此处配置仅影响FAQ模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    pretrain=dict(
        # 更多参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main pretrain --help
        train_file="input/data/faq/pre_train_data",  # 输入训练数据的路径
        vocab_file="input/data/faq/vocab",  # 模型词汇表文件的路径
        model_save_dir="models/FAQ/pretrain_model",  # 保存输出预训练checkpoint的路径
        batch_size=64,  # 预训练batch size
        train_step=10000,  # 预训练步数
        warmup_step=5000,  # 优化器warmup步数
        learning_rate=3e-5,  # 优化器学习率
        dropout_rate=0.5,  # dropout率
        seed=0,  # 初始化所用随机数种子
        print_step=1000,  # 打印log的间隔步数
        max_predictions_per_seq=10,  # 每个输入序列最大的输出长度
        weight_decay=0,  # 权重衰减系数
        clip_norm=1,  # 梯度裁剪系数
        max_seq_len=128,  # 最大输入序列长度
        init_checkpoint_file="",  # 训练起点checkpoint文件，为空时从初始状态开始训练
    ),
    finetune=dict(
        # 更多参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main finetune --help
        train_file="input/data/faq/no_commas_large_neg2pos_1/train_data",  # 输入微调数据的路径
        dev_file="input/data/faq/no_commas_large_neg2pos_1/dev_data",  # 训练时测试的数据路径
        vocab_file="input/data/faq/vocab",  # 模型词汇表文件路径
        output_id2label_file="models/FAQ/finetune_model/id2label.has_init",  # 输出id2label文件路径
        model_save_dir="models/FAQ/finetune_model",  # 保存输出模型的路径
        opt_type="adam",  # 优化器类型
        learning_rate=1e-4,  # 优化器学习率
        dropout_rate=0.1,  # dropout率
        seed=1,  # 初始化所用随机数种子
        print_step=1000,  # 打印log的间隔步数
        init_checkpoint_file="models/FAQ/pretrain_model/lm_pretrain.ckpt-10000",  # 训练起点checkpoint文件
        batch_size=64,  # 微调 batch size
        epoch=20,  # 微调时的迭代数
        max_len=128,  # 最大输入序列长度
    ),
    eval=dict(
        # 更多参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main eval --help
        input_file="input/data/faq/test_data",  # 输入测试数据的路径
        vob_file="input/data/faq/vocab",  # 模型词汇表文件的路径
        model_dir="models/FAQ/finetune_model/",  # 模型目录
        output_file="models/FAQ/result",  # 预测结果输出文件
        id2label_file="models/FAQ/finetune_model/id2label.has_init",  # 训练时生成的id2label文件路径
    ),
)


# 以下为模块配置，仅在系统运行时有效，与模型训练与测试过程无关
fengxian_qa_config = dict(
    # FengxianQA配置
    verbose=True,  # 是否输出fengxian_qa过程中的信息
    # 如希望关闭所有信息输出，请将此处，kbqa_runner_config和faq_runner_config的verbose全部设为False
)

kbqa_runner_config = dict(
    # KBQA配置
    verbose=True,  # 是否输出KBQA过程详细信息
    sim_accept_threshold=0.01,  # SIM模型认定有答案的最低信心值
    entity_linking_method="fuzzy", # 实体链接方法，可选"fuzzy"（使用EL模型）或"naive"（使用字符串匹配）

    ner=dict(
        max_seq_len=128,  # 最大输入序列长度（建议与ner_model_config相同）
        config_file='input/pretrained_BERT/bert-base-chinese-config.json',  # BERT预训练模型配置文件路径
        pre_train_model_file='models/NER/ner_output/best_ner.bin',  # 训练好的实体识别模型路径
    ),

    sim=dict(
        max_seq_len=128,  # 最大输入序列长度（建议与sim_model_config相同）
        config_file='input/pretrained_BERT/bert-base-chinese-config.json',  # BERT预训练模型配置文件路径
        pre_train_model_file='models/SIM/sim_output/best_sim.bin',  # 训练好的句子分类模型路径
    ),

    el=dict(
        pre_train_model_file='models/EL/el_output/best_el.bin',
        w2v_load_path="models/EL/el_output/w2v_model.bin",
    )
)

faq_runner_config = dict(
    # FAQ配置
    conn=dict(
        user="faq",  # mysql用户名
        host="localhost",  # mysql地址
        passwd="123456",  # mysql密码
        db="qa100",  # mysql数据库名
        charset="utf8mb4",  # mysql字符集
    ),
    verbose=True,  # 是否启用FAQ输出
    admit_threshold=0.3,  # 使用FAQ回答的最低FAQ信心值
    table_name="t_nlp_qa_faq",  # mysql表名
    vocab_file="input/data/faq/vocab",  # FAQ词汇文件路径
    model_dir="models/FAQ/finetune_model/",  # 训练好的模型路径
    id2label_file="models/FAQ/finetune_model/id2label.has_init",  # FAQ id2label文件路径
    test=dict(
        accept_thresholds=["any"],
        sim_test_file="input/data/sim/test.txt",
        faq_test_file="input/data/faq/no_commas_large_neg2pos_1/test_data",
        n_mixed_inputs=9999999,
    )
)
