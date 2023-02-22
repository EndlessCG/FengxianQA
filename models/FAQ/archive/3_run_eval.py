# -*- coding: UTF-8 -*-
"""
predict with finetuned model with testset
"""

import sys
import tensorflow as tf
import numpy as np
import argparse
import util
import datetime
import time

def get_output(g):
    return {"softmax": g.get_tensor_by_name("softmax_pre:0")}


def get_input(g):
    return {"tokens": g.get_tensor_by_name("ph_tokens:0"),
            "length": g.get_tensor_by_name("ph_length:0"),
            "dropout_rate": g.get_tensor_by_name("ph_dropout_rate:0"),
            "input_mask": g.get_tensor_by_name("ph_input_mask:0")}



def evaluate(path):
    label = []
    ans = []
    with open(path, "r") as f:
        for i in f.readlines():
            label.append(int(i.split("\t")[0].strip()))
            ans.append(int(i.split("\t")[-1].split(":")[0].strip()))
    from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
    print("准确率：", accuracy_score(label, ans))
    print("f1：", f1_score(label, ans, average='macro'))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/test_data", help="Input file for prediction.")
    parser.add_argument("--vocab_file", type=str, default="data/vocab", help="Input train file.")
    parser.add_argument("--model_path", type=str, default="", help="Path to model file.")
    parser.add_argument("--model_dir", type=str, default="finetune_model", help="Directory which contains model.")
    parser.add_argument("--output_file", type=str, default="results/result")
    parser.add_argument("--id2label_file", type=str, default="finetune_model/id2label.has_init", help="File containing (id, class label) map.")

    args = parser.parse_args()

    word2id, id2word = util.load_vocab_file(args.vocab_file)
    sys.stderr.write("vocab num : " + str(len(word2id)) + "\n")

    sens = util.gen_test_data(args.input_file, word2id)
    sys.stderr.write("sens num : " + str(len(sens)) + "\n")

    id2label = util.load_id2label_file(args.id2label_file)
    sys.stderr.write('label num : ' + str(len(id2label)) + "\n")

    # use latest checkpoint
    if "" == args.model_path:
        args.model_path = tf.train.latest_checkpoint(checkpoint_dir=args.model_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    out_list = []
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(args.model_path))
        saver.restore(sess, args.model_path)

        graph = tf.get_default_graph()
        input_dict = get_input(graph)
        output_dict = get_output(graph)

        for nu, sen in enumerate(sens):
            re = sess.run(output_dict['softmax'], feed_dict={input_dict['tokens']: [sen[0]], input_dict['input_mask']: [sen[1]], input_dict['length']: [len(sen[0])], input_dict["dropout_rate"]: 0.0})
            sorted_idx = np.argsort(-1 * re[0])  # sort by desc
            s = ""
            for i in sorted_idx[:3]:
                    s += id2label[i] + "|" + str(re[0][i]) + ","
            out_list.append(s + "\t" + " ".join([id2word[t] for t in sen[0]]) + "\n")
    # 真实标签（来自 test）
    import codecs
    real_labels = []
    for line in codecs.open(args.input_file, encoding='utf-8'):
        if len(line.split('\t')) == 3:
            real_labels.append(line.split('\t')[0])
    fout = codecs.open(args.output_file, encoding='utf-8', mode='w+')

    for idx, line in enumerate(out_list):
        line = line.strip()
        s_line = line.split('\t')
        if len(s_line) >= 2:
            model_res = s_line[0]
            sentence = s_line[1]

            model_res = model_res.replace("__label__", "") \
                .replace('|', ":").replace(",", " ")
            # real_label\tsentence\tmodel_labels
            fout.write("{}\t{}\t{}\n".format(real_labels[idx],
                                                sentence, model_res))

    fout.close()
    evaluate(args.output_file)
    


if __name__ == "__main__":
    
    tf.app.run()
    