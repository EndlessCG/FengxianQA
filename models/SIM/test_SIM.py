from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from .SIM_main import SimProcessor,SimInputFeatures,cal_acc
from utils import KBQA_TOKEN_LIST
from config import sim_model_config

import torch
import time
import numpy as np
from tqdm import tqdm, trange

config = sim_model_config.get("test", dict())
do_split_tests = config.get("do_split_tests", False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = SimProcessor()
tokenizer_inputs = ()
tokenizer_kwards = {'do_lower_case': False,
                    'max_len': 50,
                    'vocab_file': 'input/pretrained_BERT/bert-base-chinese-vocab.txt'}
tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)
tokenizer.add_special_tokens(KBQA_TOKEN_LIST)

features_dict = dict()
# if do_split_tests:
#     for t_type in ["1hop"]:
#         features_dict[t_type] = torch.load(f'input/data/sim/cached_test_{t_type}_50')
features_dict["all SIM"] = torch.load('input/data/sim/cached_test_128')

def do_test(test_type, features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    test_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)


    bert_config = BertConfig.from_pretrained('input/pretrained_BERT/bert-base-chinese-config.json')
    bert_config.num_labels = len(processor.get_labels())

    model = BertForSequenceClassification(bert_config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(sim_model_config.get("test", dict()).get("model_path")))
    model = model.to(device)


    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,batch_size=256)

    total_loss = 0.       # loss 的总和
    total_sample_num = 0  # 样本总数目
    all_real_label = []   # 记录所有的真实标签列表
    pred_logits = []   # 记录所有的预测标签列表
    times = []

    for batch in tqdm(test_dataloader, desc="testing"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3],
                    }
            start = time.time()
            outputs = model(**inputs)
            end = time.time()
            times.append(end - start)
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
    print(f"response time: {np.average(times)} +- {np.std(times)}")
    print("loss",loss.item())
    print("question_acc",question_acc)
    print("label_acc",label_acc)

def main():
    for t_type, features in features_dict.items():
        do_test(t_type, features)

if __name__ == '__main__':
    main()