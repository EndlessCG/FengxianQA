from .BERT_CRF import BertCrf
from transformers import BertTokenizer
from .NER_main import NerProcessor,statistical_real_sentences,flatten,CrfInputFeatures
from config import ner_model_config
from torch.utils.data import DataLoader, RandomSampler,TensorDataset
from sklearn.metrics import classification_report
import torch
import numpy as np
import time
from tqdm import tqdm, trange

config = ner_model_config.get("test", dict())
do_split_tests = config.get("do_split_tests", False)

processor = NerProcessor()
tokenizer_inputs = ()
tokenizer_kwards = {'do_lower_case': False,
                    'max_len': 50,
                    'vocab_file': 'input/pretrained_BERT/bert-base-chinese-vocab.txt'}
tokenizer = BertTokenizer(*tokenizer_inputs,**tokenizer_kwards)

model = BertCrf(config_name= 'input/pretrained_BERT/bert-base-chinese-config.json',
                num_tags = len(processor.get_labels()),batch_first=True)
model.load_state_dict(torch.load(ner_model_config.get("test", dict()).get("model_path")))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

features_dict = dict()
# if do_split_tests:
#     for t_type in ["1hop"]:
#     # for t_type in ["mhop"]:
#         features_dict[t_type] = torch.load(f'input/data/fengxian/ner/cached_test_{t_type}_50')
features_dict["all NER"] = torch.load('input/data/fengxian/ner/cached_test_50')

def do_test(test_type, features, duplicate=10):
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
    times = []

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
            start = time.time()
            outputs = model(**inputs)
            end = time.time()
            # temp_eval_loss shape: (batch_size)
            # temp_pred : list[list[int]] 长度不齐
            temp_eval_loss, temp_pred = outputs[0], outputs[1]
            times.append(end - start)
            loss.extend(temp_eval_loss.tolist())
            pred_token_label.extend(temp_pred)
            real_token_label.extend(statistical_real_sentences(batch[3], batch[1], temp_pred))

    loss = np.array(loss).mean()
    # real_token_label = np.array(flatten(real_token_label))
    # pred_token_label = np.array(flatten(pred_token_label))
    # print(real_token_label)
    # assert real_token_label.shape == pred_token_label.shape
    # ret = classification_report(y_true=real_token_label, y_pred=pred_token_label, digits = 6,output_dict=False)
    acc = sum([1 if pred_token_label[i] == real_token_label[i] else 0 for i in range(len(pred_token_label))]) / len(pred_token_label)
    print(f"TEST TYPE:\t{test_type}")
    print(f"response time: {np.average(times[1:])} +- {np.std(times[1:])}")
    print(f"Problem ACCURACY:\t{acc}")

def main():
    for t_type, features in features_dict.items():
        do_test(t_type, features)

if __name__ == '__main__':
    main()
