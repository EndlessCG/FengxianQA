import torch
import time
import os
import argparse
import numpy as np
from tqdm import tqdm, trange
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils.operations import convert_config_paths, merge_arg_and_config

from .SIM_main import SimProcessor,SimInputFeatures,cal_acc
from utils import KBQA_TOKEN_LIST
from config import sim_model_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_sim_test(args):
    features_dict = dict()
    if args.do_split_tests:
        for t_type in ["1hop", "mhop", "unchain1hop", "unchainmhop"]:
            features_dict[t_type] = torch.load(f'input/data/sim/cached_test_{t_type}_{args.max_seq_length}')
    features_dict["all SIM"] = torch.load(os.path.join(args.data_dir, f"cached_test_{args.max_seq_length}"))
    return features_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/SIM/sim_output/best_sim.bin", help="待测试模型路径")
    parser.add_argument("--data_dir", default="input/data/sim/", help="训练过程中生成的cached测试数据所在文件夹")
    parser.add_argument("--vob_file", default='input/pretrained_BERT/bert-base-chinese-vocab.txt', help="待测试模型词汇表路径")
    parser.add_argument("--model_config_file", default="input/pretrained_BERT/bert-base-chinese-config.json", help="待测试模型配置文件路径")
    parser.add_argument("--max_seq_length", default=128, help="最大序列长度")
    parser.add_argument("--do_split_tests", default=True, help="是否进行分类型问题测试")
    args = parser.parse_args()
    config = sim_model_config.get("test", dict())
    convert_config_paths(config)
    merge_arg_and_config(args, config)
    return args


def do_test(args, test_type, features):
    tokenizer_inputs = ()
    tokenizer_kwards = {'do_lower_case': False,
                        'max_len': args.max_seq_length,
                        'vocab_file': args.vob_file}
    tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)
    tokenizer.add_special_tokens(KBQA_TOKEN_LIST)

    processor = SimProcessor()
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    test_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)


    bert_config = BertConfig.from_pretrained(args.model_config_file)
    bert_config.num_labels = len(processor.get_labels())

    model = BertForSequenceClassification(bert_config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)


    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,batch_size=256)

    total_loss = 0.       # loss 的总和
    total_sample_num = 0  # 样本总数目
    all_real_label = []   # 记录所有的真实标签列表
    pred_logits = []   # 记录所有的预测标签列表
    for batch in tqdm(test_dataloader, desc="testing"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3],
                    }
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            total_loss += loss * batch[0].shape[0]  # loss * 样本个数
            total_sample_num += batch[0].shape[0]  # 记录样本个数

            # pred = logits.argmax(dim=-1).tolist()  # 得到预测的label转为list

            # all_pred_label.extend(pred)  # 记录预测的 label
            pred_logits.extend(logits.softmax(dim=-1).tolist())
            all_real_label.extend(batch[3].view(-1).tolist())  # 记录真实的label
    loss = total_loss / total_sample_num
    question_acc,label_acc = cal_acc(pred_logits, all_real_label)

    print(f"TEST TYPE:\t{test_type}")
    print("loss",loss.item())
    print("question_acc",question_acc)
    print("label_acc",label_acc)

def main():
    args = parse_args()
    features_dict = init_sim_test(args)
    for t_type, features in features_dict.items():
        do_test(args, t_type, features)

if __name__ == '__main__':
    main()