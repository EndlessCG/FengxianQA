from torch.utils.data import DataLoader, RandomSampler,TensorDataset
import torch
import numpy as np
import time
import os
import argparse
from tqdm import tqdm, trange

from utils.operations import convert_config_paths, merge_arg_and_config
from .BERT_CRF import BertCrf
from .NER_main import NerProcessor,statistical_real_sentences,flatten,CRF_LABELS,CrfInputFeatures
from config import ner_model_config

temp_output_file = 'temp_ner_test_result.txt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def init_ner_test(args):
    processor = NerProcessor()

    model = BertCrf(config_name= args.model_config_file,
                    num_tags = len(processor.get_labels()),
                    batch_first=True)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    features_dict = dict()
    raw_input_path_dict = dict()
    do_split_tests = args.do_split_tests
    if do_split_tests:
        for t_type in ["1hop"]:
        # for t_type in ["mhop"]:
            features_dict[t_type] = torch.load(f'input/data/ner/cached_test_{t_type}_50')
    features_dict["all NER"] = torch.load(os.path.join(args.data_dir, f"cached_test_{args.max_seq_length}"))
    raw_input_path_dict["all NER"] = args.raw_data_path
    return model, features_dict, raw_input_path_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/NER/ner_output/best_ner.bin", help="待测试模型路径")
    parser.add_argument("--data_dir", default="input/data/ner/", help="训练过程中生成的cached测试数据所在文件夹")
    parser.add_argument("--raw_data_path", default="input/data/ner/test.txt", help="测试数据原文件路径")
    parser.add_argument("--vob_file", default='input/pretrained_BERT/bert-base-chinese-vocab.txt', help="待测试模型词汇表路径")
    parser.add_argument("--model_config_file", default="input/pretrained_BERT/bert-base-chinese-config.json", help="待测试模型配置文件路径")
    parser.add_argument("--max_seq_length", default=128, help="最大序列长度")
    args = parser.parse_args()
    config = ner_model_config.get("test", dict())
    convert_config_paths(config)
    merge_arg_and_config(args, config)
    return args
    

def do_test(model, raw_input_path, test_type, features, duplicate=10):
    all_input_ids = torch.tensor([f.input_ids for f in features] * duplicate, dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features] * duplicate, dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features] * duplicate, dtype=torch.long)
    all_label = torch.tensor([f.label for f in features] * duplicate, dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)

    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=256)
    loss = []
    real_token_label = []
    pred_token_label = []

    for batch in tqdm(data_loader, desc="test"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'tags': batch[3],
                    'decode': True,
                    'reduction': 'none'
                    }
            outputs = model(**inputs)
            # temp_eval_loss shape: (batch_size)
            # temp_pred : list[list[int]] 长度不齐
            temp_eval_loss, temp_pred = outputs[0], outputs[1]
            loss.extend(temp_eval_loss.tolist())
            pred_token_label.extend(temp_pred)
            real_token_label.extend(statistical_real_sentences(batch[3], batch[1], temp_pred))

    loss = np.array(loss).mean()
    real_input_list = []
    with open(raw_input_path) as f:
        for real_input_line in f.readlines():
            real_input_line_list = real_input_line.split(' ')
            if len(real_input_line_list) == 2:
                real_input_list.append(real_input_line_list[0])
    with open(temp_output_file, 'w+') as f:
        for pred_token, real_token, raw_input in zip(flatten(pred_token_label), flatten(real_token_label), real_input_list):
            f.write(f"{raw_input} {CRF_LABELS[real_token]} {CRF_LABELS[pred_token]}\n")

    print(f"TEST TYPE:\t{test_type}")
    os.system(f"python utils/conlleval.py {temp_output_file}")
    os.system(f"rm {temp_output_file}")

def main():
    args = parse_args()
    model, features_dict, raw_input_path_dict = init_ner_test(args)
    for t_type, features in features_dict.items():
        raw_input_path = raw_input_path_dict[t_type]
        do_test(model, raw_input_path, t_type, features)

if __name__ == '__main__':
    main()
