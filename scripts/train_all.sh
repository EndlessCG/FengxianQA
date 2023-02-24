#! /bin/bash
bert_kbqa_home=$(cd $(dirname $0); cd ..; pwd)

# train FAQ
echo "Training FAQ..."
FAQ_home=$bert_kbqa_home/models/FAQ/
cd $FAQ_home
python data/process/process_std.py
python data/process/data_generate.py
python data/process/file_to_mysql.py
python FAQ_main.py pretrain --batch_size 16
python FAQ_main.py finetune --batch_size 16
echo "Testing FAQ..."
python FAQ_main.py eval

# train NER
cd $bert_kbqa_home
echo $bert_kbqa_home
if [ ! -d "models/input/data/fengxian/" ]; then
    python preprocess/data_generator.py
fi
echo "Training NER..."
python -m models.NER.NER_main --train_batch_size 8 --eval_batch_size 8
echo "Testing NER..."
python -m models.NER.test_NER

# train SIM
echo "Training SIM..."
python -m models.SIM.SIM_main --train_batch_size 8 --eval_batch_size 8
echo "Testing SIM..."
python -m models.SIM.test_SIM
