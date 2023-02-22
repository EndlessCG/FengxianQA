bert_kbqa_config = dict(
    neo4j=dict(
        neo4j_addr="bolt://localhost:7687",
        username="neo4j",
        password="123456"
    ),
    ner=dict(
        config_file='models/input/config/bert-base-chinese-config.json',
        pre_train_model='models/ner_output/best_ner.bin',
    ),
    sim=dict(
        config_file='models/input/config/bert-base-chinese-config.json',
        pre_train_model='models/sim_output/best_sim.bin'
    ),
)

faq_config = dict(
    sql_addr="faq",
    sql_host="localhost",
    sql_password="123456",
    sql_database="qa100",
    sql_table="qa100",
    input_file="models/FAQ/data/ans_data",
    vocab_file="models/FAQ/data/vocab",
    model_dir="models/FAQ/finetune_model/",
    output_file="models/FAQ/results/result",
    id2label_file="models/FAQ/finetune_model/id2label.has_init"
)
