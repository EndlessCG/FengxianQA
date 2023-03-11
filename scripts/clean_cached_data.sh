#! /bin/bash
bert_kbqa_home=$(cd $(dirname $0); cd ..; pwd)

cd $bert_kbqa_home
rm input/data/ner/cached* || true
rm input/data/sim/cached* || true
