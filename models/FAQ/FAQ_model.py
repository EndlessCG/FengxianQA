from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import pymysql
from . import models_util
from . import util
import gc
import time
import sys

class FAQ():
    def __init__(self, init_args=None):
        self._model_loaded = False
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        tf.logging.set_verbosity(tf.logging.INFO)
        if init_args is not None:
            self.initialize(init_args)
    
    def initialize(self, args):
        if hasattr(args, "conn"):
            self.sql_conn = pymysql.connect(**args.conn)
        else:
            self.sql_conn = pymysql.connect(
                host=args.sql_host,
                user=args.sql_user,
                passwd=args.sql_passwd,
                charset=args.sql_charset,
                db=args.sql_db,
            )
        self.table_name = args.table_name
        self.word2id, self.id2word = util.load_vocab_file(args.vocab_file)
        self.id2label = util.load_id2label_file(args.id2label_file)


    def pretrain(self, args):
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        tf.logging.info(args)
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)

        # load data
        word2id, id2word = util.load_vocab_file(args.vocab_file)
        training_sens = util.load_pretraining_data(args.train_file, args.max_seq_len)

        if not args.use_queue:
            util.to_ids(training_sens, word2id, args, id2word)

        other_arg_dict = {}
        other_arg_dict['token_num'] = len(word2id)

        # load model 
        model = models_util.create_bidirectional_lm_training_op(args, other_arg_dict)

        gc.collect()
        saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            if args.init_checkpoint_file:
                tf.logging.info('restore the checkpoint : ' + str(args.init_checkpoint_file))
                saver.restore(sess, args.init_checkpoint_file)

            total_loss = 0
            num = 0
            global_step = 0
            while global_step < args.train_step:
                if not args.use_queue:
                    iterator = util.gen_batches(training_sens, args.batch_size)
                else:
                    iterator = util.queue_gen_batches(training_sens, args, word2id, id2word)
                assert iterator is not None
                for batch_data in iterator:
                    feed_dict = {model.ph_tokens: batch_data[0],
                                model.ph_length: batch_data[1],
                                model.ph_labels: batch_data[2],
                                model.ph_positions: batch_data[3],
                                model.ph_weights: batch_data[4],
                                model.ph_input_mask: batch_data[5],
                                model.ph_dropout_rate: args.dropout_rate}
                    _, global_step, loss, learning_rate = sess.run([model.train_op, \
                                                                    model.global_step, model.loss_op,
                                                                    model.learning_rate_op], feed_dict=feed_dict)

                    total_loss += loss
                    num += 1
                    if global_step % args.print_step == 0:
                        tf.logging.info("\nglobal step : " + str(global_step) +
                                        ", avg loss so far : " + str(total_loss / num) +
                                        ", instant loss : " + str(loss) +
                                        ", learning_rate : " + str(learning_rate) +
                                        ", time :" + str(time.strftime('%Y-%m-%d %H:%M:%S')))
                        tf.logging.info("save model ...")
                        saver.save(sess, args.model_save_dir + '/lm_pretrain.ckpt', global_step=global_step)
                        gc.collect()

                if not args.use_queue:
                    util.to_ids(training_sens, word2id, args, id2word)  # MUST run this for randomization for each sentence
                gc.collect()

    def _finetune_eval(self, sess, full_tensors, args, model):
        total_num = 0
        right_num = 0
        for batch_data in util.gen_batchs(full_tensors, args.batch_size, is_shuffle=False):
            softmax_re = sess.run(model.softmax_op,
                                feed_dict={model.ph_dropout_rate: 0,
                                            model.ph_tokens: batch_data[0],
                                            model.ph_labels: batch_data[1],
                                            model.ph_length: batch_data[2],
                                            model.ph_input_mask: batch_data[3]})
            pred_re = np.argmax(softmax_re, axis=1)
            total_num += len(pred_re)
            right_num += np.sum(pred_re == batch_data[1])
            acc = 1.0 * right_num / (total_num + 1e-5)

        tf.logging.info("dev total num: " + str(total_num) + ", right num: " + str(right_num) + ", acc: " + str(acc))
        return acc
    
    def finetune(self, args):
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        tf.logging.info(str(args))
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)

        tf.logging.info("load training sens")
        train_sens = util.load_training_data(args.train_file, skip_invalid=True)
        tf.logging.info("\nload dev sens")
        dev_sens = util.load_training_data(args.dev_file, skip_invalid=True)

        word2id, id2word, label2id, id2label = util.load_vocab(train_sens + dev_sens, args.vocab_file)
        fw = open(args.output_id2label_file, 'w+')
        for k, v in id2label.items():
            fw.write(str(k) + "\t" + v + "\n")
        fw.close()

        util.gen_ids(train_sens, word2id, label2id, args.max_len)
        util.gen_ids(dev_sens, word2id, label2id, args.max_len)

        train_full_tensors = util.make_full_tensors(train_sens)
        dev_full_tensors = util.make_full_tensors(dev_sens)

        other_arg_dict = {}
        other_arg_dict['token_num'] = len(word2id)
        other_arg_dict['label_num'] = len(label2id)
        model = models_util.create_finetune_classification_training_op(args, other_arg_dict)

        steps_in_epoch = int(len(train_sens) // args.batch_size)
        tf.logging.info("batch size: " + str(args.batch_size) + ", training sample num : " + str(
            len(train_sens)) + ", print step : " + str(args.print_step))
        tf.logging.info(
            "steps_in_epoch : " + str(steps_in_epoch) + ", epoch num :" + str(args.epoch) + ", total steps : " + str(
                args.epoch * steps_in_epoch))
        print_step = min(args.print_step, steps_in_epoch)
        tf.logging.info("eval dev every {} step".format(print_step))

        save_vars = [v for v in tf.global_variables() if
                    v.name.find('adam') < 0 and v.name.find('Adam') < 0 and v.name.find('ADAM') < 0]
        tf.logging.info(str(save_vars))
        tf.logging.info(str(tf.all_variables()))

        saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            total_loss = 0
            dev_best_so_far = 0
            for epoch in range(1, args.epoch + 1):
                tf.logging.info("\n" + "*" * 20 + "epoch num :" + str(epoch) + "*" * 20)
                for batch_data in util.gen_batchs(train_full_tensors, args.batch_size, is_shuffle=True):
                    _, global_step, loss = sess.run([model.train_op, model.global_step_op, model.loss_op],
                                                    feed_dict={model.ph_dropout_rate: args.dropout_rate,
                                                            model.ph_tokens: batch_data[0],
                                                            model.ph_labels: batch_data[1],
                                                            model.ph_length: batch_data[2],
                                                            model.ph_input_mask: batch_data[3]})
                    total_loss += loss
                    if global_step % print_step == 0:
                        tf.logging.info(
                            "\nglobal step : " + str(global_step) + ", avg loss so far : " + str(total_loss / global_step))
                        tf.logging.info("begin to eval dev set: ")
                        acc = self._finetune_eval(sess, dev_full_tensors, args, model)
                        if acc > dev_best_so_far:
                            dev_best_so_far = acc
                            tf.logging.info("!" * 20 + "best got : " + str(acc))
                            # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["scores"])
                            saver.save(sess, args.model_save_dir + '/finetune.ckpt', global_step=global_step)

                tf.logging.info("\n----------------------eval after one epoch: ")
                tf.logging.info(
                    "global step : " + str(global_step) + ", avg loss so far : " + str(total_loss / global_step))
                tf.logging.info("begin to eval dev set: ")
                sys.stdout.flush()
                acc = self._finetune_eval(sess, dev_full_tensors, args, model)
                if acc > dev_best_so_far:
                    dev_best_so_far = acc
                    tf.logging.info("!" * 20 + "best got : " + str(acc))
                    saver.save(sess, args.model_save_dir + '/finetune.ckpt', global_step=global_step)
    
    def _get_output(self, g):
        return {"softmax": g.get_tensor_by_name("softmax_pre:0")}

    def _get_input(self, g):
        return {"tokens": g.get_tensor_by_name("ph_tokens:0"),
                "length": g.get_tensor_by_name("ph_length:0"),
                "dropout_rate": g.get_tensor_by_name("ph_dropout_rate:0"),
                "input_mask": g.get_tensor_by_name("ph_input_mask:0")}

    def _print_metrics(self, path):
        label = []
        ans = []
        with open(path, "r") as f:
            for i in f.readlines():
                label.append(int(i.split("\t")[0].strip()))
                ans.append(int(i.split("\t")[-1].split(":")[0].strip()))
        from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
        print("准确率：", accuracy_score(label, ans))
        print("f1：", f1_score(label, ans, average='macro'))

    def _get_answer_by_id(self, pred_id):
        id2answer_query = f"SELECT answer from {self.table_name} WHERE `id`={pred_id}"
        cursor = self.sql_conn.cursor()
        cursor.execute(id2answer_query)
        answer = cursor.fetchall()[0][0]
        return answer


    def evaluate_model(self, args):
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
            input_dict = self._get_input(graph)
            output_dict = self._get_output(graph)

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
        self._print_metrics(args.output_file)
    
    def predict(self, args, input_sentence=""):
        if input_sentence == "" and hasattr(args, 'input_sentence'):
            input_sentence = args.input_sentence

        sens = util.get_single_data(input_sentence, self.word2id)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        out_list = []
        if not self._model_loaded:
            self.load_model(model_path=args.model_path)

        if hasattr(args, "profile_faq"):
            output_file = args.profile_faq
            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        for sen in sens:
            re = self.session.run(self.output_dict['softmax'], feed_dict={self.input_dict['tokens']: [sen[0]],
                                                                  self.input_dict['input_mask']: [sen[1]],
                                                                  self.input_dict['length']: [len(sen[0])],
                                                                  self.input_dict["dropout_rate"]: 0.0},
                                                                  options=run_options,
                                                                  run_metadata=run_metadata)
            if hasattr(args, "profile_faq"):
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(output_file, 'w') as f:
                    f.write(ctf)
            
            sorted_idx = np.argsort(-1 * re[0])  # sort by desc
            s = ""
            for i in sorted_idx[:3]:
                s += self.id2label[i] + "|" + str(re[0][i]) + ","
            out_list.append(s + "\t" + " ".join([self.id2word[t] for t in sen[0]]) + "\n")

        for idx, line in enumerate(out_list):
            line = line.strip()
            s_line = line.split('\t')
            if len(s_line) >= 2:
                model_res = s_line[0]
                sentence = s_line[1]

                model_res = model_res.replace("__label__", "") \
                    .replace('|', ":").replace(",", " ")
                # real_label\tsentence\tmodel_labels
                fir_ans_num = int(model_res.split(":")[0].strip())
                fir_ans_prob = model_res.split(":")[1].split()[0].strip()
                # print(fir_ans_prob)
                # print("问题：{}".format(args.input_sentence))
                answer = self._get_answer_by_id(fir_ans_num)
                if len(answer) < 1 or answer == "nan":
                    return "No FAQ answer", 0
                else:
                    return f"{answer}", float(fir_ans_prob) # 输出

    def load_model(self, model_path="", model_dir=""):
        if "" == model_path:
            model_path = tf.train.latest_checkpoint(checkpoint_dir=model_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(model_path))
        saver.restore(self.session, model_path)

        graph = tf.get_default_graph()
        self.input_dict = self._get_input(graph)
        self.output_dict = self._get_output(graph)
        self._model_loaded = True

    def destroy(self):
        self.session.close()
        self._model_loaded = False
