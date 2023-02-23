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
echo "Training NER..."
NER_home=$bert_kbqa_home/models/NER/
cd $NER_home
python NER_main.py
echo "Testing FAQ..."
python test_NER.py

# train SIM
echo "Training SIM..."
SIM_home=$bert_kbqa_home/models/SIM/
cd $SIM_home
python SIM_main.py
echo "Testing FAQ..."
python test_SIM.py
