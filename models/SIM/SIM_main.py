# --data_dir
# ./input/data/sim
# --vob_file
# ./input/pretrained_BERT/bert-base-chinese-vocab.txt
# --model_config
# ./input/pretrained_BERT/bert-base-chinese-config.json
# --output
# ./output
# --pre_train_model
# ./input/pretrained_BERT/bert-base-chinese-model.bin
# --max_seq_length
# 64
# --do_train
# --train_batch_size
# 32
# --eval_batch_size
# 256
# --gradient_accumulation_steps
# 4
# --num_train_epochs
# 15

import argparse
from collections import Counter
import code
import os
import logging
from tqdm import tqdm, trange
import random
import codecs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import DataProcessor, InputExample

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from utils import KBQA_TOKEN_LIST, merge_arg_and_config, convert_config_paths
from config import sim_model_config, kbqa_runner_config
logger = logging.getLogger(__name__)



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def cal_acc(pred_logits, real_label):
    real_label = torch.tensor(real_label)
    pred_logits = torch.tensor(pred_logits)
    pred_label = pred_logits.argmax(dim=-1)

    assert real_label.shape == pred_label.shape
    # assert 0 == real_label.shape[0] % split_size

    label_acc = (real_label == pred_label).sum().float() / float(pred_label.shape[0])
    
    reshaped = False
    split_size = np.where(real_label == 1)[0][1]
    if 0 == real_label.shape[0] % split_size:
        reshaped = True
        real_label = real_label.reshape(-1, split_size)
    
    if reshaped and real_label.shape[0] == real_label[:,0].sum():
        pred_logits = pred_logits[:, 1].reshape(-1, split_size)
        pred_values, pred_idx = pred_logits.max(dim=-1)
        # 可能性最大的问题==0且可能性>0.1
        question_acc = ((pred_idx == 0) & (pred_values > kbqa_runner_config.get("sim_accept_threshold", 0.1))).sum().float() / float(pred_idx.shape[0])
        return question_acc.item(),label_acc.item()
    else:
        print("SIM cal_acc: pos & neg samples in dataset not formally aligned, disabling question_acc")
        return -1, label_acc.item()
    # 测试用的
    # return ((real_label.argmax(dim=-1) == 0).sum() / real_label.shape[0]).item()





class SimInputExample(object):
    def __init__(self, guid, question,attribute, label=None):
        self.guid = guid
        self.question = question
        self.attribute = attribute
        self.label = label


class SimInputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class SimProcessor(DataProcessor):
    """Processor for the FAQ problem
        modified from https://github.com/huggingface/transformers/blob/master/transformers/data/processors/glue.py#L154
    """

    def get_train_examples(self, data_dir):
        logger.info("*******  train  ********")
        return self._create_examples(
            os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        logger.info("*******  dev  ********")
        return self._create_examples(
            os.path.join(data_dir, "validate.txt"))

    def get_test_examples(self,data_dir, subtest=""):
        logger.info(f"*******  test {subtest} ********")
        file_path = "test.txt" if subtest == "" else f"{subtest}.txt"
        return self._create_examples(
            os.path.join(data_dir, file_path))

    def get_labels(self):
        return [0, 1]

    @classmethod
    def _create_examples(cls, path):
        examples = []
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split('\t')
                if 4 == len(tokens):
                    examples.append(
                        SimInputExample(guid=int(tokens[0]),
                                        question=tokens[1],
                                        attribute=tokens[2],
                                        label=int(tokens[3]))
                    )
        f.close()
        return examples


def sim_convert_examples_to_features(examples,tokenizer,
                                     max_length=512,
                                     label_list=None,
                                     pad_token=0,
                                     pad_token_segment_id = 0,
                                     mask_padding_with_zero = True):

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            text = example.question,
            text_pair= example.attribute,
            add_special_tokens=True,
            max_length=max_length
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)


        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),max_length)

        # label = label_map[example.label]
        label = example.label


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % str(label))

        features.append(
            SimInputFeatures(input_ids,attention_mask,token_type_ids,label)
        )
    return features


def load_and_cache_example(args,tokenizer,processor,data_type):

    type_list = ['train','validate','test', 'test_1hop', 'test_mhop', 'test_unchain1hop', 'test_unchainmhop']
    if data_type not in type_list:
        raise ValueError("data_type must be one of {}".format(" ".join(type_list)))

    cached_features_file = "cached_{}_{}".format(data_type,str(args.max_seq_length))
    cached_features_file = os.path.join(args.data_dir,cached_features_file)
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if type_list[0] == data_type:
            examples = processor.get_train_examples(args.data_dir)
        elif type_list[1] == data_type:
            examples = processor.get_dev_examples(args.data_dir)
        elif type_list[2] == data_type:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir, subtest=data_type)

        features = sim_convert_examples_to_features(examples=examples,tokenizer=tokenizer,max_length=args.max_seq_length,label_list=label_list)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    return dataset


def trains(args,train_dataset,eval_dataset,model):

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight','transitions']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    best_acc = 0.
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step,batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'labels':batch[3],
            }
            outputs = model(**inputs)
            loss,logits = outputs[0], outputs[1]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            logging_loss += loss.item()
            tr_loss += loss.item()
            if 0 == (step + 1) % args.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                logger.info("EPOCH = [%d/%d] global_step = %d   loss = %f",_+1,args.num_train_epochs,global_step,
                            logging_loss)
                logging_loss = 0.0

                # if (global_step < 100 and global_step % 10 == 0) or (global_step % 50 == 0):
                # 每 相隔 100步，评估一次
                if (global_step % 50 == 0 and global_step <= 100) or(global_step % 100 == 0 and global_step < 1000) \
                     or (global_step % 200 == 0):

                    best_acc = evaluate_and_save_model(args,model,eval_dataset,_,global_step,best_acc)
    #
    # # 最后循环结束 再评估一次
    best_acc = evaluate_and_save_model(args,model,eval_dataset,_,global_step,best_acc)



def evaluate_and_save_model(args,model,eval_dataset,epoch,global_step,best_acc):
    eval_loss, question_acc,label_acc = evaluate(args, model, eval_dataset)
    logger.info("Evaluating EPOCH = [%d/%d] global_step = %d eval_loss = %f question_acc = %f label_acc = %f",
                epoch + 1, args.num_train_epochs,global_step,eval_loss, question_acc,label_acc)
    if question_acc > best_acc:
        best_acc = question_acc
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_sim.bin'))
        logging.info("save the best model %s , question_acc = %f",
                     os.path.join(args.output_dir, 'best_sim.bin'),best_acc)
    return best_acc



def evaluate(args, model, eval_dataset):

    eval_output_dirs = args.output_dir
    if not os.path.exists(eval_output_dirs):
        os.makedirs(eval_output_dirs)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    total_loss = 0.       # loss 的总和
    total_sample_num = 0  # 样本总数目
    all_real_label = []   # 记录所有的真实标签列表
    all_logits = []   # 记录所有的预测标签列表
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'labels':batch[3],
            }
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            total_loss += loss * batch[0].shape[0]    # loss * 样本个数
            total_sample_num += batch[0].shape[0]     # 记录样本个数

            # pred = logits.argmax(dim=-1).tolist()     # 得到预测的label转为list

            all_logits.append(logits)                        # 记录预测的 label
            all_real_label.extend(batch[3].view(-1).tolist())  # 记录真实的label

    loss = total_loss / total_sample_num
    question_acc,label_acc = cal_acc(torch.cat(all_logits).softmax(dim=-1).detach().cpu().numpy(), 
                                    all_real_label)

    model.train()
    return loss,question_acc,label_acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="input/data/sim", type=str, required=False,
                        help="数据文件目录，因当有train.text dev.text")

    parser.add_argument("--vob_file", default="input/pretrained_BERT/bert-base-chinese-vocab.txt", type=str, required=False,
                        help="词表文件")
    parser.add_argument("--model_config_file", default="input/pretrained_BERT/bert-base-chinese-config.json", type=str, required=False,
                        help="模型配置文件json文件")
    parser.add_argument("--pre_train_model_file", default="input/pretrained_BERT/bert-base-chinese-model.bin", type=str, required=False,
                        help="预训练的模型文件，参数矩阵。如果存在就加载")
    parser.add_argument("--output_dir", default="models/SIM/sim_output/", type=str, required=False,
                        help="输出的模型文件名")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="输入到bert的最大长度，通常不应该超过512")
    parser.add_argument("--do_train", action='store_true',default=True,
                        help="是否进行训练")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="训练集的batch_size")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="验证集的batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="最大的梯度更新")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="epoch 数目")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="让学习增加到1的步数，在warmup_steps后，再衰减到0")

    config = sim_model_config.get("train", dict())
    args = parser.parse_args()
    convert_config_paths(config)
    merge_arg_and_config(args, config)
    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.vob_file)
    assert os.path.exists(args.model_config_file)
    assert os.path.exists(args.pre_train_model_file)


    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    #filename = './output/bert-sim.log',


    processor = SimProcessor()
    tokenizer_inputs = ()
    tokenizer_kwards = {'do_lower_case': False,
                        'max_len': args.max_seq_length,
                        'vocab_file': args.vob_file}
    tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)
    tokenizer.add_special_tokens(KBQA_TOKEN_LIST)

    train_dataset = load_and_cache_example(args, tokenizer, processor, 'train')
    eval_dataset = load_and_cache_example(args, tokenizer, processor, 'validate')
    _ = load_and_cache_example(args,tokenizer,processor,'test')
    _ = load_and_cache_example(args,tokenizer,processor,'test_1hop')
    _ = load_and_cache_example(args,tokenizer,processor,'test_mhop')
    _ = load_and_cache_example(args,tokenizer,processor,'test_unchain1hop')
    _ = load_and_cache_example(args,tokenizer,processor,'test_unchainmhop')



    bert_config = BertConfig.from_pretrained(args.model_config_file)
    bert_config.num_labels = len(processor.get_labels()) # [0,1]
    model_kwargs = {'config':bert_config}

    model = BertForSequenceClassification.from_pretrained(args.pre_train_model_file, **model_kwargs)
    model = model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        trains(args,train_dataset,eval_dataset,model)


if __name__ == '__main__':
    main()

