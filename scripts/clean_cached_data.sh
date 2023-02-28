#! /bin/bash
bert_kbqa_home=$(cd $(dirname $0); cd ..; pwd)

cd $bert_kbqa_home
rm input/data/fengxian/ner/cached* || true
rm input/data/fengxian/sim/cached* || true
