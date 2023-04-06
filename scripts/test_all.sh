#! /bin/bash
bert_kbqa_home=$(cd $(dirname $0); cd ..; pwd)
cd $bert_kbqa_home

# model test
echo "Testing FAQ..."
python -m models.FAQ.FAQ_main eval
echo "Testing NER..."
python -m models.NER.test_NER
echo "Testing SIM..."
python -m models.SIM.test_SIM
echo "Tesing EL..."
python -m models.EL.test_EL

# runner test
echo "Testing FAQ runner..."
python -m runners.test_faq_runner
echo "Testing KBQA runner..."
python -m runners.test_kbqa_runner

# response time test
echo "Testing response time..."
python -m utils.test_response_time

# system test
echo "Testing FengxianQA..."
python -m utils.test_fengxianqa
