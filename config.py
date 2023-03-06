fengxian_qa_config = dict(
    # FengxianQA配置
    verbose=False, # 是否输出fengxian_qa过程中的信息
                   # 如希望关闭所有信息输出，请将此处，kbqa_runner_config和faq_runner_config的verbose全部设为False
)

kbqa_runner_config = dict(
    # KBQA配置
    verbose=False, # 是否输出KBQA过程详细信息
    sim_accept_threshold=0.01, # SIM模型认定有答案的最低信心值
    neo4j=dict(
        neo4j_addr="bolt://localhost:7687", # neo4j地址
        username="neo4j", # neo4j用户名
        password="123456", # neo4j密码
    ),

    ner=dict(
        config_file='input/pretrained_BERT/bert-base-chinese-config.json', # 模型config文件路径
        pre_train_model='models/NER/ner_output/best_ner.bin', # 训练过的模型的路径
    ),

    sim=dict(
        config_file='input/pretrained_BERT/bert-base-chinese-config.json', # 模型config文件路径
        pre_train_model='models/SIM/sim_output/best_sim.bin', # 训练过的模型的路径
    ),
)

faq_runner_config = dict(
    # FAQ配置
    conn=dict(
        user="faq", # mysql用户名
        host="localhost", # mysql地址
        passwd="123456", # mysql密码
        db="qa100", # mysql数据库名
        charset="utf8mb4", # mysql用字符集
    ),
    verbose=False, # 是否启用FAQ输出（暂无效果）
    admit_threshold=0.8, # 使用FAQ回答的最低FAQ信心值
    table_name="t_nlp_qa_faq", # mysql表名
    vocab_file="input/data/fengxian/faq/vocab", # FAQ词汇文件路径
    model_dir="models/FAQ/finetune_model/", # 训练好的模型路径
    id2label_file="models/FAQ/finetune_model/id2label.has_init" # FAQ id2label文件路径
)

ner_model_config=dict(

    # NER模型训练与测试配置
    # 此处配置仅影响NER模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    train=dict(
        # 参数定义请参考models/NER/NER_main.py或运行python -m models.NER.NER_main --help
        data_dir="input/data/fengxian/ner",
        vob_file="input/pretrained_BERT/bert-base-chinese-vocab.txt",
        model_config="input/pretrained_BERT/bert-base-chinese-config.json",
        output_dir="models/NER/ner_output",
        pre_train_model="input/pretrained_BERT/bert-base-chinese-model.bin",
        max_seq_length=50,
        train_batch_size=32,
        eval_batch_size=32,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
    ),
)

sim_model_config=dict(

    # SIM模型训练配置
    # 此处配置仅影响SIM模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    train=dict(
        # 参数定义请参考models/SIM/SIM_main.py或运行python -m models.SIM.SIM_main --help
        data_dir="input/data/fengxian/sim",
        vob_file="input/pretrained_BERT/bert-base-chinese-vocab.txt",
        model_config="input/pretrained_BERT/bert-base-chinese-config.json",
        output_dir="models/SIM/ner_output",
        pre_train_model="input/pretrained_BERT/bert-base-chinese-model.bin",
        max_seq_length=50,
        train_batch_size=32,
        eval_batch_size=32,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
    ),
)

faq_model_config=dict(

    # FAQ模型训练配置
    # 此处配置仅影响FAQ模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    pretrain=dict(
        # 参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main pretrain --help
        train_file="input/data/fengxian/faq/pre_train_data",
        vocab_file="input/data/fengxian/faq/vocab",
        model_save_dir="models/FAQ/pretrain_model",
    ),
    finetune=dict(
        # 参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main finetune --help
        train_file="input/data/fengxian/faq/train_data",
        dev_file="input/data/fengxian/faq/dev_data",
        vocab_file="input/data/fengxian/faq/vocab",
        output_id2label_file="models/FAQ/finetune_model/id2label.has_init",
        model_save_dir="models/FAQ/finetune_model",
        batch_size=32,
        epoch=30,
    ),
    eval=dict(
        # 参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main eval --help
        
    ),
    predict=dict(
        # 参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main predict --help
    ),
)

ner_model_config=dict(

    # NER模型训练与测试配置
    # 此处配置仅影响NER模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    train=dict(
        # 参数定义请参考models/NER/NER_main.py或运行python -m models.NER.NER_main --help
        data_dir="input/data/fengxian/ner",
        vob_file="input/pretrained_BERT/bert-base-chinese-vocab.txt",
        model_config="input/pretrained_BERT/bert-base-chinese-config.json",
        output_dir="models/NER/ner_output",
        pre_train_model="input/pretrained_BERT/bert-base-chinese-model.bin",
        max_seq_length=50,
        train_batch_size=32,
        eval_batch_size=32,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
    ),
    test=dict(
        model_path='models/NER/ner_output/best_ner.bin',
        do_split_tests=True,
    )
)

sim_model_config=dict(

    # SIM模型训练配置
    # 此处配置仅影响SIM模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    train=dict(
        # 参数定义请参考models/SIM/SIM_main.py或运行python -m models.SIM.SIM_main --help
        data_dir="input/data/fengxian/sim",
        vob_file="input/pretrained_BERT/bert-base-chinese-vocab.txt",
        model_config="input/pretrained_BERT/bert-base-chinese-config.json",
        output_dir="models/SIM/ner_output",
        output_model_name="best_sim_neg_to_pos_3.bin",
        pre_train_model="input/pretrained_BERT/bert-base-chinese-model.bin",
        max_seq_length=50,
        train_batch_size=32,
        eval_batch_size=32,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
    ),
    test=dict(
        model_path='models/SIM/sim_output/best_sim.bin',
        do_split_tests=True,
    )
)

faq_model_config=dict(

    # FAQ模型训练配置
    # 此处配置仅影响FAQ模型训练与测试过程，与FengxianQA推理（即do_qa）过程无关
    pretrain=dict(
        # 参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main pretrain --help
        train_file="input/data/fengxian/faq/pre_train_data",
        vocab_file="input/data/fengxian/faq/vocab",
        model_save_dir="models/FAQ/pretrain_model",
    ),
    finetune=dict(
        # 参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main finetune --help
        train_file="input/data/fengxian/faq/train_data",
        dev_file="input/data/fengxian/faq/dev_data",
        vocab_file="input/data/fengxian/faq/vocab",
        output_id2label_file="models/FAQ/finetune_model/id2label.has_init",
        model_save_dir="models/FAQ/finetune_model",
        batch_size=32,
        epoch=30,
    ),
    eval=dict(
        # 参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main eval --help
        
    ),
    predict=dict(
        # 参数定义请参考models/FAQ/FAQ_main.py或运行python -m models.FAQ.FAQ_main predict --help
        
    ),
)