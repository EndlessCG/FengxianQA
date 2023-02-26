bert_kbqa_config = dict(
    neo4j=dict(
        neo4j_addr="bolt://localhost:7687",
        username="neo4j",
        password="123456"
    ),
    ner=dict(
        config_file='input/pretrained_BERT/bert-base-chinese-config.json',
        pre_train_model='models/NER/ner_output/best_ner.bin',
    ),
    sim=dict(
        config_file='input/pretrained_BERT/bert-base-chinese-config.json',
        pre_train_model='models/SIM/sim_output/best_sim.bin'
    ),
)

faq_config = dict(
    conn=dict(
        user="faq",
        host="localhost",
        passwd="123456",
        db="qa100",
        charset="utf8mb4",
    ),
    admit_threshold=0.8,
    table_name="t_nlp_qa_faq",
    vocab_file="input/data/fengxian/faq/vocab",
    model_dir="models/FAQ/finetune_model/",
    output_file="models/FAQ/result",
    id2label_file="models/FAQ/finetune_model/id2label.has_init"
)
