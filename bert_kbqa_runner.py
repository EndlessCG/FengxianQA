from models.BERT_CRF import BertCrf
from models.NER_main import NerProcessor, CRF_LABELS
from models.SIM_main import SimProcessor, SimInputFeatures
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from question_intents import QUESTION_INTENTS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
# import pymysql
from tqdm import tqdm, trange
from utils.neo4j_graph import Neo4jGraph


class BertKBQARunner():
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('连接数据库...')
        neo4j_config = config['neo4j']
        self.graph = Neo4jGraph(neo4j_config['neo4j_addr'], 
                                neo4j_config['username'], 
                                neo4j_config['password'])
        
        print('加载模型...')
        with torch.no_grad():
            tokenizer_inputs = ()
            tokenizer_kwards = {'do_lower_case': False,
                                'max_len': 30,
                                'vocab_file': 'models/input/config/bert-base-chinese-vocab.txt'}
            self.ner_processor = NerProcessor()
            self.sim_processor = SimProcessor()
            self.tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)

            self.ner_model = self.get_ner_model(config_file='models/input/config/bert-base-chinese-config.json',
                                           pre_train_model='models/ner_output/best_ner.bin',
                                           label_num=len(self.ner_processor.get_labels()))
            self.ner_model = self.ner_model.to(self.device)
            self.ner_model.eval()

            self.sim_model = self.get_sim_model(config_file='models/input/config/bert-base-chinese-config.json',
                                      pre_train_model='models/sim_output/best_sim.bin',
                                      label_num=len(self.sim_processor.get_labels()))

            self.sim_model = self.sim_model.to(self.device)
            self.sim_model.eval()


    def get_ner_model(self, config_file, pre_train_model, label_num=2):
        model = BertCrf(config_name=config_file,
                        num_tags=label_num, batch_first=True)
        model.load_state_dict(torch.load(pre_train_model, map_location="cpu"))
        return model.to(self.device)


    def get_sim_model(self, config_file, pre_train_model, label_num=2):
        bert_config = BertConfig.from_pretrained(config_file)
        bert_config.num_labels = label_num
        model = BertForSequenceClassification(bert_config)
        model.load_state_dict(torch.load(pre_train_model, map_location="cpu"))
        return model.to(self.device)


    def get_entity(self, sentence, max_len=64):
        model = self.ner_model
        tokenizer = self.tokenizer
        pad_token = 0
        sentence_list = list(sentence.strip().replace(' ', ''))
        text = " ".join(sentence_list)
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        labels_ids = None

        assert len(input_ids) == max_len, "Error with input length {} vs {}".format(
            len(input_ids), max_len)
        assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(
            len(attention_mask), max_len)
        assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(
            len(token_type_ids), max_len)

        input_ids = torch.tensor(input_ids).reshape(1, -1).to(self.device)
        attention_mask = torch.tensor(
            attention_mask).reshape(1, -1).to(self.device)
        token_type_ids = torch.tensor(
            token_type_ids).reshape(1, -1).to(self.device)
        labels_ids = labels_ids

        model = model.to(self.device)
        model.eval()
        # 由于传入的tag为None，所以返回的loss 也是None
        ret = model(input_ids=input_ids,
                    tags=labels_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
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
        for i in range(entity_start_idx+1, pre_tag_len):
            if pre_tag[i] == i_loc_idx:
                entity_list.append(sentence_list[i])
            else:
                break
        return "".join(entity_list)


    def semantic_matching(self, question, attribute_list, max_length):
        model = self.sim_model
        tokenizer = self.tokenizer
        pad_token = 0
        pad_token_segment_id = 1
        features = []
        for (ex_index, attribute) in enumerate(attribute_list):
            inputs = tokenizer.encode_plus(
                text=question,
                text_pair=attribute,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            attention_mask = [1] * len(input_ids)

            padding_length = max_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + \
                ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
                len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_length)
            features.append(
                SimInputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            )
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long)

        assert all_input_ids.shape == all_attention_mask.shape
        assert all_attention_mask.shape == all_token_type_ids.shape

        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=128)

        data_num = all_attention_mask.shape[0]
        batch_size = 128

        all_logits = None
        for i in range(0, data_num, batch_size):
            model.eval()
            with torch.no_grad():
                inputs = {'input_ids': all_input_ids[i:i+batch_size].to(self.device),
                        'attention_mask': all_attention_mask[i:i+batch_size].to(self.device),
                        'token_type_ids': all_token_type_ids[i:i+batch_size].to(self.device),
                        'labels': None
                        }
                outputs = model(**inputs)
                logits = outputs[0]
                logits = logits.softmax(dim=-1)

                if all_logits is None:
                    all_logits = logits.clone()
                else:
                    all_logits = torch.cat([all_logits, logits], dim=0)
        pre_rest = all_logits.argmax(dim=-1)
        if 0 == pre_rest.sum():
            return torch.tensor(-1)
        else:
            return pre_rest.argmax(dim=-1)

    def do_qa(self, question):
        graph = self.graph
        # 1. Mention Recognition
        mention_list = []
        # entity list matching
        for e in graph.entity_list:
            if e in question:
                mention_list.append(e)
                break
        # NER
        ner_mention = self.get_entity(sentence=question, max_len=30)
        if ner_mention != '':
            mention_list.append(ner_mention)

        mention_list = set(mention_list)
        if len(mention_list) != 0:
            print("候选实体：", mention_list)
        else: # 如果问题中不存在实体
            print("回答：未找到该问题中的实体")

        # 2. Entity Linking
        for mention in mention_list:
            if mention in graph.entity_list:
                # 完全匹配
                entity = mention
            else:
                # 局部匹配
                entity = next((e for e in graph.entity_list if mention in e), None)
            if entity is not None:
                break
        
        if entity is None:
            # 未找到
            print(f"回答：未找到\"{mention}\"相关信息")
            return
        
        print("链接到的实体：", entity)

        # 3. Intention Mapping + Attribute/Relation Retrival
        # 获取实体对应的所有属性
        get_e_relation_query = f"match (n)-[r]-() where n.`名称`='{entity}' return type(r)"
        get_e_attribute_query =  f"match (n) where n.`名称` = '{entity}' return keys(n)"
        e_relations = graph.execute_query(get_e_relation_query)
        e_attributes = graph.execute_query(get_e_attribute_query)[0]
        relations, attributes = [], []

        match_idx = self.semantic_matching(question, e_relations + e_attributes, 30).item()
        if match_idx == -1:
            print(f"回答：未在\"{entity}\"中找到问题相关信息")
            return
        elif match_idx < len(e_relations):
            intention = 'one_hop_e'
            relations = [e_relations[match_idx]]
        else:
            intention = 'one_hop_a'
            attributes = [e_attributes[match_idx - len(e_relations)]]

        # 4. Query Generation
        answer_query = QUESTION_INTENTS[intention]['query']
        slots = QUESTION_INTENTS[intention]['query_slots']
        slot_fills = []
        for slot_type, slot_idx in slots:
            if slot_type == 'e':
                slot_fills.append(entity)
            elif slot_type == 'a':
                slot_fills.append(attributes[slot_idx])
            elif slot_type == 'r':
                slot_fills.append(relations[slot_idx])
        answer_query = answer_query.format(*slot_fills)
        values = graph.execute_query(answer_query)
        
        # 5. Answer Generation
        answer_template = QUESTION_INTENTS[intention]['answer']
        slots = QUESTION_INTENTS[intention]['answer_slots']
        slot_fills = []
        for slot_type, slot_idx in slots:
            if slot_type == 'e':
                slot_fills.append(entity)
            elif slot_type == 'a':
                slot_fills.append(attributes[slot_idx])
            elif slot_type == 'r':
                slot_fills.append(relations[slot_idx])
            elif slot_type == 'v':
                slot_fills.append('，'.join(values))
        answer = answer_template.format(*slot_fills)
        print("回答：", answer)