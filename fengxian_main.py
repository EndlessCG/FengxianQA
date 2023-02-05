from BERT_CRF import BertCrf
from NER_main import NerProcessor, CRF_LABELS
from SIM_main import SimProcessor,SimInputFeatures
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
# import pymysql
from tqdm import tqdm, trange
from py2neo import Graph, Node, NodeMatcher
graph = Graph("bolt://localhost:7687", auth=('neo4j', '123456'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

def get_ner_model(config_file,pre_train_model,label_num = 2):
    model = BertCrf(config_name=config_file,num_tags=label_num, batch_first=True)
    model.load_state_dict(torch.load(pre_train_model,map_location="cpu"))
    return model.to(device)


def get_sim_model(config_file,pre_train_model,label_num = 2):
    bert_config = BertConfig.from_pretrained(config_file)
    bert_config.num_labels = label_num
    model = BertForSequenceClassification(bert_config)
    model.load_state_dict(torch.load(pre_train_model,map_location="cpu"))
    return model


def get_entity(model,tokenizer,sentence,max_len = 64):
    pad_token = 0
    sentence_list = list(sentence.strip().replace(' ',''))
    text = " ".join(sentence_list)
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        truncate_first_sequence=True  # We're truncating the first sequence in priority if True
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)
    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    labels_ids = None

    assert len(input_ids) == max_len, "Error with input length {} vs {}".format(len(input_ids), max_len)
    assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(len(attention_mask), max_len)
    assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(len(token_type_ids), max_len)

    input_ids = torch.tensor(input_ids).reshape(1,-1).to(device)
    attention_mask = torch.tensor(attention_mask).reshape(1,-1).to(device)
    token_type_ids = torch.tensor(token_type_ids).reshape(1,-1).to(device)
    labels_ids = labels_ids

    model = model.to(device)
    model.eval()
    # 由于传入的tag为None，所以返回的loss 也是None
    ret = model(input_ids = input_ids,
                  tags = labels_ids,
                  attention_mask = attention_mask,
                  token_type_ids = token_type_ids)
    pre_tag = ret[1][0]
    assert len(pre_tag) == len(sentence_list) or len(pre_tag) == max_len - 2

    pre_tag_len = len(pre_tag)
    b_loc_idx = CRF_LABELS.index('B-entity')
    i_loc_idx = CRF_LABELS.index('I-entity')
    o_idx = CRF_LABELS.index('O')

    if b_loc_idx not in pre_tag and i_loc_idx not in pre_tag:
        print("没有在句子[{}]中发现实体".format(sentence))
        return ''
    if b_loc_idx in pre_tag:

        entity_start_idx = pre_tag.index(b_loc_idx)
    else:

        entity_start_idx = pre_tag.index(i_loc_idx)
    entity_list = []
    entity_list.append(sentence_list[entity_start_idx])
    for i in range(entity_start_idx+1,pre_tag_len):
        if pre_tag[i] == i_loc_idx:
            entity_list.append(sentence_list[i])
        else:
            break
    return "".join(entity_list)


def semantic_matching(model,tokenizer,question,attribute_list,max_length):

    pad_token = 0
    pad_token_segment_id = 1
    features = []
    for (ex_index, attribute) in enumerate(attribute_list):
        inputs = tokenizer.encode_plus(
            text = question,
            text_pair = attribute,
            add_special_tokens = True,
            max_length = max_length,
            truncate_first_sequence = True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)
        features.append(
            SimInputFeatures(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    assert all_input_ids.shape == all_attention_mask.shape
    assert all_attention_mask.shape == all_token_type_ids.shape


    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,batch_size=128)

    data_num = all_attention_mask.shape[0]
    batch_size = 128

    all_logits = None
    for i in range(0,data_num,batch_size):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': all_input_ids[i:i+batch_size].to(device),
                      'attention_mask': all_attention_mask[i:i+batch_size].to(device),
                      'token_type_ids': all_token_type_ids[i:i+batch_size].to(device),
                      'labels': None
                      }
            outputs = model(**inputs)
            logits = outputs[0]
            logits = logits.softmax(dim = -1)

            if all_logits is None:
                all_logits = logits.clone()
            else:
                all_logits = torch.cat([all_logits,logits],dim = 0)
    pre_rest = all_logits.argmax(dim = -1)
    if 0 == pre_rest.sum():
        return torch.tensor(-1)
    else:
        return pre_rest.argmax(dim = -1)


def main():

    with torch.no_grad():
        tokenizer_inputs = ()
        tokenizer_kwards = {'do_lower_case': False,
                            'max_len': 30,
                            'vocab_file': 'input/config/bert-base-chinese-vocab.txt'}
        ner_processor = NerProcessor()
        sim_processor = SimProcessor()
        tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)


        ner_model = get_ner_model(config_file = 'input/config/bert-base-chinese-config.json',
                                  pre_train_model = 'ner_output/best_ner.bin',label_num = len(ner_processor.get_labels()))
        ner_model = ner_model.to(device)
        ner_model.eval()

        sim_model = get_sim_model(config_file='./input/config/bert-base-chinese-config.json',
                                  pre_train_model='sim_output/best_sim.bin',
                                  label_num=len(sim_processor.get_labels()))

        sim_model = sim_model.to(device)
        sim_model.eval()

        # while True:
        #     print("====="*10)
        #     raw_text = input("问题：") 
        #     raw_text = raw_text.strip() # 去掉输入的首尾空格
        #     if ( "quit" == raw_text ):
        #         print("quit")
        #         return
        raw_text = "客户管理是啥意思？他涉及哪些部分？"
        entity = get_entity(model=ner_model, tokenizer=tokenizer, sentence=raw_text, max_len=30)
        # print(type(entity))
        print("问题中的实体：", entity)
        if '' == entity: # 如果问题中不存在实体
            print("回答：未找到该问题的答案")
            # continue #记得取消注释
        else: # 补全局部匹配
            flag = 0 # 0代表完全匹配
            sql_str = "match (n) where n.`名称` = '{}' return keys(n)".format(entity) 
            attributes = graph.run(sql_str).data()
            if len(attributes)== 0: # 如果完全匹配没有检索到实体，调用局部匹配
                sql_str = "match (n) where n.`名称` contains('{}') return keys(n)".format(entity) 
                attributes = graph.run(sql_str).data()
                flag =1 # 1代表局部匹配
                
            if len(attributes)== 0: # 如果完全匹配没有检索到实体，局部匹配也没有检索到，直接输出答案
                print("回答：未找到该问题的答案")
            else:
                # 获取实体对应的所有属性
                attribute_list = []
                for attribute in attributes:
                    attribute_list+=list(attribute.values())[0]
                
                # 获取实体对应的所有关系
                if flag ==0: 
                    sql_str = "match (n)-[r]-() where n.`名称` = '{}' return type(r)".format(entity) 
                else:
                    sql_str = "match (n)-[r]-() where n.`名称` contains('{}') return type(r)".format(entity) 
                relationships = graph.run(sql_str).data()
                for relationship in relationships: 
                    attribute_list.append(relationship["type(r)"])
                    
                attribute_list =list(set(attribute_list))
                attribute_idx = semantic_matching(sim_model, tokenizer, raw_text, attribute_list, 30).item()
                if -1 == attribute_idx:
                    ret = ''
                else:
                    attribute = attribute_list[attribute_idx] # 提取的问题中的属性
                    print("属性：",attribute)
                    
                    # 获取属性对应的值【答案】
                    sql_attribute = ""
                    if flag ==0:
                        sql_attribute = "match (n) where n.`名称` = '{}' return n.`{}`".format(entity,attribute) 
                    else:
                        sql_attribute = "match (n) where n.`名称` contains('{}') return n.`{}`".format(entity,attribute) 
                    results_attribute = graph.run(sql_attribute).data()
                    
                    # 获取关系对应的节点【答案】
                    sql_relationship =""
                    if flag ==0:
                        sql_relationship = "match (n)-[r]-(m) where n.`名称` = '{}' and type(r)= '{}' return m.`名称`".format(entity,attribute) 
                    else:
                        sql_relationship = "match (n)-[r]-(m) where n.`名称` contains('{}') and type(r)= '{}' return m.`名称`".format(entity,attribute) 
                    results_relationship = graph.run(sql_relationship).data()
                    
                    answer = []
                    if len(results_attribute)!=0:
                        for result in results_attribute:
                            if result["n.`{}`".format(attribute)] is not None:
                                answer.append(result["n.`{}`".format(attribute)])
                                
                    if len(results_relationship)!=0:
                        for result in results_relationship:
                            if result["m.`名称`"] is not None:
                                answer.append(result["m.`名称`"])
                    
                    ret = "{}的{}是{}".format(entity, attribute, ",".join(answer))
                    
                if '' == ret:
                    print("属性：无")
                    print("回答：未找到该问题的答案")
                else:
                    print("回答：",ret)

if __name__ == '__main__':
    main()















