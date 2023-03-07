from models.NER.BERT_CRF import BertCrf
from models.NER.NER_main import NerProcessor, CRF_LABELS
from models.SIM.SIM_main import SimProcessor, SimInputFeatures
import transformers
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from utils.question_intents import QUESTION_INTENTS, SUBGRAPHS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import pandas as pd
# import pymysql
from tqdm import tqdm, trange
from itertools import chain
from utils import Neo4jGraph, KBQA_TOKEN_LIST, get_abs_path


class BertKBQARunner():
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transformers.logging.set_verbosity_error()
        self.config = config
        self._verbose = config.get("verbose", "False")
        self._print('连接数据库...')
        neo4j_config, ner_config, sim_config = config['neo4j'], config['ner'], config['sim']
        self.graph = Neo4jGraph(neo4j_config['neo4j_addr'], 
                                neo4j_config['username'], 
                                neo4j_config['password'])
        
        with torch.no_grad():
            self._load_ner_model(ner_config)
            self._load_sim_model(sim_config)
            
    def _load_ner_model(self, ner_config):
        self._print('加载NER模型...')
        tokenizer_inputs = ()
        tokenizer_kwards = {'do_lower_case': False,
                            'max_len': 40,
                            'vocab_file': get_abs_path('input/pretrained_BERT/bert-base-chinese-vocab.txt')}
        self.ner_processor = NerProcessor()
        self.sim_processor = SimProcessor()
        self.tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)
        self.tokenizer.add_special_tokens(KBQA_TOKEN_LIST)

        self.ner_model = self.get_ner_model(config_file=ner_config.get('config_file', 'input/pretrained_BERT/bert-base-chinese-config.json'),
                                        pre_train_model=ner_config.get('pre_train_model_file','models/ner_output/best_ner.bin'),
                                        label_num=len(self.ner_processor.get_labels()))
        self.ner_model = self.ner_model.to(self.device)
        self.ner_model.eval()

    def _load_sim_model(self, sim_config):
        self._print('加载SIM模型...')
        self.sim_model = self.get_sim_model(config_file=sim_config.get('config_file', 'input/pretrained_BERT/bert-base-chinese-config.json'),
                                        pre_train_model=sim_config.get('pre_train_model_file', 'sim_output/best_sim.bin'),
                                        label_num=len(self.sim_processor.get_labels()))
        self.sim_model = self.sim_model.to(self.device)
        self.sim_model.eval()

    def _print(self, *args):
        if self._verbose:
            print("KBQA:", *args)

    def get_ner_model(self, config_file, pre_train_model, label_num=2):
        model = BertCrf(config_name=config_file,
                        num_tags=label_num, batch_first=True)
        model.load_state_dict(torch.load(pre_train_model, map_location="cpu"))
        return model.to(self.device)


    def get_sim_model(self, config_file, pre_train_model, label_num=2):
        bert_config = BertConfig.from_pretrained(config_file)
        bert_config.num_labels = label_num
        model = BertForSequenceClassification(bert_config)
        model.resize_token_embeddings(len(self.tokenizer))
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
        b_entity_idx = CRF_LABELS.index('B-entity')
        i_entity_idx = CRF_LABELS.index('I-entity')
        b_attr_idx = CRF_LABELS.index('B-attribute')
        i_attr_idx = CRF_LABELS.index('B-attribute')
        o_idx = CRF_LABELS.index('O')

        if not any(i in pre_tag for i in [b_entity_idx, i_entity_idx, b_attr_idx, i_attr_idx]):
            self._print("没有在句子[{}]中发现实体".format(sentence))
            return '', ''
        
        entity_start_idx, attr_start_idx = -1, -1
        if b_entity_idx in pre_tag:
            entity_start_idx = pre_tag.index(b_entity_idx)
        elif b_attr_idx in pre_tag:
            attr_start_idx = pre_tag.index(b_attr_idx)
        elif i_entity_idx in pre_tag:
            entity_start_idx = pre_tag.index(i_entity_idx)
        elif i_attr_idx in pre_tag:
            attr_start_idx = pre_tag.index(b_attr_idx)
        
        entity, attribute = "", ""
        if entity_start_idx != -1:    
            entity += sentence_list[entity_start_idx]
            for i in range(entity_start_idx + 1, pre_tag_len):
                if pre_tag[i] == i_entity_idx:
                    entity += sentence_list[i]
                else:
                    break
        
        if attr_start_idx != -1:
            attribute += sentence_list[attr_start_idx]
            for i in range(attr_start_idx + 1, pre_tag_len):
                if pre_tag[i] == i_attr_idx:
                    attribute += sentence_list[i]
                else:
                    break
        
        return entity, attribute


    def semantic_matching(self, question, attribute_list, max_length, top_k=1):
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
        # pre_rest = all_logits.argmax(dim=-1)
        if self.config.get("sim_accept_threshold", 0.01) > all_logits[:,1].max(dim=0)[0]:
            return torch.tensor(-1)
        else:
            return torch.topk(all_logits[:,1], k=top_k).indices

    def subgraph_generation(self):
        self.subgraph_candidates = []
        for query in SUBGRAPHS.values():
            query_result = self.graph.execute_query(query)
            self.subgraph_candidates.extend(query_result)

    def do_qa(self, question):
        graph = self.graph
        # 1. Mention Recognition
        e_mention_list = []
        a_mention_list = []
        # entity list matching
        for e in graph.entity_list:
            if e in question:
                e_mention_list.append(e)
                break
        # NER
        ner_e_mention, ner_a_mention = self.get_entity(sentence=question, max_len=40)
        if ner_e_mention != '':
            e_mention_list.append(ner_e_mention)
        if ner_a_mention != '':
            a_mention_list.append(ner_a_mention)

        e_mention_list = set(e_mention_list)
        a_mention_list = set(a_mention_list)
        if len(e_mention_list) != 0:
            self._print("候选实体：", e_mention_list)
        if len(a_mention_list) != 0:
            self._print("候选属性：", a_mention_list)
        if len(e_mention_list) == 0 and len(a_mention_list) == 0: # 如果问题中不存在实体
            return "未找到该问题中的实体"

        # 2. Entity Linking
        linked_entity, linked_attribute = [], []
        for mention in e_mention_list:
            if mention in graph.entity_list:
                # 完全匹配
                linked_entity.append(mention)
            else:
                # 局部匹配
                entity = next((e for e in graph.entity_list if mention in e), None)
                if entity is not None and not entity in linked_entity:
                    linked_entity.append(entity)
        
        for mention in a_mention_list:
            if mention in graph.attribute_list:
                # 完全匹配
                linked_attribute.append(mention)
            else:
                # 局部匹配
                attribute = next((e for e in graph.entity_list if mention in e), None)
                if attribute is not None and not attribute in linked_attribute:
                    linked_attribute.append(attribute)
            
        if len(linked_attribute) == 0 and len(linked_entity) == 0:
            # 未找到
            return f"未找到\"{e_mention_list.union(a_mention_list)}\"相关信息"
        
        self._print("链接到的实体：", linked_entity)
        self._print("链接到的属性：", linked_attribute)

        # 3. Candidate Subgraph Generation
        sgraph_type_idx = {}
        sgraph_candidates = []
        acc_idx = 0
        query_slots = {}
        for i, e in enumerate(linked_entity):
            query_slots["entity" + ("" if i == 0 else str(i))] = e
        for i, a in enumerate(linked_attribute):
            query_slots["attribute" + ("" if i == 0 else str(i))] = a
        # naive subgraph generation (without pruning)
        for sgraph_type, query in SUBGRAPHS.items():
            try:
                query_result = self.graph.execute_query(query.format(**query_slots))
            except KeyError:
                continue
            sgraph_candidates += query_result
            for g in query_result:
                sgraph_type_idx[g] = sgraph_type
            acc_idx += len(query_result)
    
        # 4. Cadidate Subgraph Selection
        max_sgraph_len = max([len(sg) for sg in sgraph_candidates])
        match_idx = self.semantic_matching(question, sgraph_candidates, max_sgraph_len).item()
        if match_idx == -1:
            return f"未在\"{','.join(linked_entity + linked_attribute)}\"中找到问题相关信息"
        
        intention = sgraph_type_idx[sgraph_candidates[match_idx]]
        self._print(f"问题类型：{QUESTION_INTENTS[intention].get('display', intention)}")
        self._print(f"问题路径：{sgraph_candidates[match_idx]}")

        # 5. Query Generation
        answer_query = QUESTION_INTENTS[intention]['query']
        slots = QUESTION_INTENTS[intention]['query_slots']
        slot_fills = []

        temp_sgraph = sgraph_candidates[match_idx].replace('[TARGET]', '[NEDGE]')
        links = temp_sgraph.split('[NEDGE]')
        while '' in links:
            links.remove('')
        for slot_type, slot_idx in slots:
            if slot_type == 'e':
                slot_fills.append(linked_entity[slot_idx])
            elif slot_type == 'a':
                slot_fills.append(linked_attribute[slot_idx])
            elif slot_type == 'l':
                slot_fills.append(links[slot_idx])
        answer_query = answer_query.format(*slot_fills)
        values = graph.execute_query(answer_query)
        if values == [None] or any([v is None for v in chain.from_iterable(values)]):
            return f"未找到问题相关信息"
        
        # 6. Answer Generation
        answer_templates = QUESTION_INTENTS[intention]['answer']
        slots = QUESTION_INTENTS[intention]['answer_slots']
        answer_list = []
        if isinstance(answer_templates, str):
            answer_templates = [answer_templates]
        if isinstance(slots[0], tuple):
            slots = [slots]
        for answer_template, slot in zip(answer_templates, slots):
            slot_fills = []
            v_cnt = 0
            for i, (slot_type, slot_idx) in enumerate(slot):
                if slot_type == 'e':
                    slot_fills.append([linked_entity[slot_idx]])
                elif slot_type == 'a':
                    slot_fills.append([linked_attribute[slot_idx]])
                elif slot_type == 'l':
                    slot_fills.append([links[slot_idx]])
                elif slot_type == 'v':
                    if len(values) != 0 and isinstance(values[0], list):
                        values_slot_idx = [v_row[slot_idx] for v_row in values]
                    else:
                        values_slot_idx = values
                    v_cnt += 1
                    slot_fills.append([v[:-1] if v[-1] == '。' else v for v in values_slot_idx])
            
            if v_cnt > 1:
                slot_fills_df = pd.DataFrame(slot_fills).T.fillna(method='pad')
                answer_list.extend([answer_template.format(*row) for _, row in slot_fills_df.iterrows()])
            else:
                answer_list.append(answer_template.format(*["，".join(s) for s in slot_fills]))
        
        return "。".join(answer_list) + "。"
