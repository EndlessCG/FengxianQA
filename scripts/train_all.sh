#! /bin/bash
bert_kbqa_home=$(cd $(dirname $0); cd ..; pwd)

# train FAQ
# echo "Training FAQ..."
# if [ ! -f "input/data/fengxian/faq/train_data" ]; then
#     python preprocess/FAQ/process_std.py
#     python preprocess/FAQ/data_generate.py
#     python preprocess/FAQ/file_to_mysql.py
# fi
# python -m models.FAQ.FAQ_main pretrain
# python -m models.FAQ.FAQ_main finetune

# train NER
cd $bert_kbqa_home
bash scripts/clean_cached_data.sh
if [ ! -d "input/data/fengxian/ner" ]; then
    mkdir -p input/data/fengxian/ner/
    mkdir -p input/data/fengxian/sim/
    python preprocess/data_generator.py
fi
cd $bert_kbqa_home
config_size=$(wc -c <"input/pretrained_BERT/bert-base-chinese-config.json")
if [ ! -f "input/pretrained_BERT/bert-base-chinese-model.bin" ] || 
   [ ! -f "input/pretrained_BERT/bert-base-chinese-config.json" ] || 
   [ ! -f "input/pretrained_BERT/bert-base-chinese-vocab.txt" ] ||
   [ $config_size -lt 10 ]; then
    echo "Downloading bert-base-chinese..."
    mkdir -p input/pretrained_BERT/
    cd input/pretrained_BERT
    wget https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin -O bert-base-chinese-model.bin
    wget https://huggingface.co/bert-base-chinese/resolve/main/config.json -O bert-base-chinese-config.json
    wget https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt -O bert-base-chinese-vocab.txt
    cd $bert_kbqa_home
fi
echo "Training NER..."
python -m models.NER.NER_main

# train SIM
echo "Training SIM..."
python -m models.SIM.SIM_main

# test
echo "Testing FAQ..."
python -m models.FAQ.FAQ_main eval
echo "Testing NER..."
python -m models.NER.test_NER
echo "Testing SIM..."
python -m models.SIM.test_SIM