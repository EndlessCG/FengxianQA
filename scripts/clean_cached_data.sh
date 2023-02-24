#! /bin/bash
bert_kbqa_home=$(cd $(dirname $0); cd ..; pwd)

cd $bert_kbqa_home
rm models/input/data/fengxian/ner/cached*
rm models/input/data/fengxian/sim_data/cached*
