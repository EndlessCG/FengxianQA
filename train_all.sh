#! /bin/bash
bert_kbqa_home=$(dirname $0)

# train FAQ
echo "Training FAQ..."
FAQ_home=$bert_kbqa_home/models/FAQ/
cd $FAQ_home
python data/process/process_std.py
python data/process/data_generate.py
python data/process/file_to_mysql.py
python FAQ_main.py pretrain
python FAQ_main.py finetune
echo "Testing FAQ..."
python FAQ_main.py evaluate

# train NER
cd $bert_kbqa_home
echo "Training NER..."
python -m models.NER.NER_main
echo "Testing FAQ..."
python -m models.NER.test_NER

# train SIM
echo "Training SIM..."
python -m models.SIM.SIM_main
echo "Testing FAQ..."
python -m models.SIM.test_SIM
