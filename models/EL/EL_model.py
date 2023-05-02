# word2vec dataset: https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d100-v0.2.0-s.tar.gz
from gensim.models import KeyedVectors
from gensim.summarization.bm25 import BM25
import jieba
import pickle
import argparse
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBRanker
import numpy as np
from sklearn.metrics import f1_score
from Levenshtein import distance as lev

class EL():
    def __init__(self, config, neo4j_graph):
        if isinstance(config, argparse.Namespace):
            self.args = config
        else:
            self.args = argparse.Namespace(**config)
            self._verbose = config.get("verbose", "True")
        print("Loading w2v model...This might take several minutes...")
        if hasattr(self.args, "w2v_load_path") and osp.exists(self.args.w2v_load_path):
            self.w2v_model = pickle.load(open(self.args.w2v_load_path, 'rb'))
        else:
            self.w2v_model = KeyedVectors.load_word2vec_format(self.args.w2v_corpus_path, binary=False)
            if not os.path.exists(osp.dirname(self.args.w2v_load_path)):
                os.makedirs(osp.dirname(self.args.w2v_load_path))
            pickle.dump(self.w2v_model, open(self.args.w2v_load_path, 'wb'), protocol=4)
        self.graph = neo4j_graph
        self.full_entity_list = self.graph.entity_list + self.graph.attribute_list
        print("Tokenizing entity list...")
        tok_entity_list = [jieba.lcut(entity) for entity in self.full_entity_list]
        print("Building BM25...")
        self.bm25_model = BM25(tok_entity_list)
        xgb.set_config(verbosity=2)
        self.el_model = XGBRanker(tree_method='gpu_hist')
    
    def _print(self, args):
        if self._verbose:
            print("EL:", *args)

    def _string_similarity(self, mention, entity):
        # Jaccard similarity
        tok_mention = jieba.lcut(mention)
        tok_entity = jieba.lcut(entity)
        intersection = set(tok_mention) & set(tok_entity)
        union = set(tok_mention) | set(tok_entity)
        return len(intersection) / len(union)
    
    def _word2vec_similarity(self, s1, s2):
        cut_s1, cleaned_s1 = jieba.lcut(s1.replace(' ', '')), []
        cut_s2, cleaned_s2 = jieba.lcut(s2.replace(' ', '')), []
        for w1 in cut_s1:
            if w1 not in self.w2v_model:
                cleaned_s1.extend([char for char in w1 if char in self.w2v_model])
            else:
                cleaned_s1.append(w1)
        for w2 in cut_s2:
            if w2 not in self.w2v_model:
                cleaned_s2.extend([char for char in w2 if char in self.w2v_model])
            else:
                cleaned_s2.append(w2)
        if len(cleaned_s1) == 0 or len(cleaned_s2) == 0:
            return 0
        return self.w2v_model.n_similarity(cleaned_s1, cleaned_s2)

    def _entity_popularity(self, entity):
        pop_query = f"match (n) where n.`名称`='{entity}' with size((n)-[]-()) as degree return degree;"
        result = self.graph.execute_query(pop_query)
        return 0 if result == [] else result[0]

    def _levenshstein_distance(self, s1, s2):
        return lev(s1, s2)

    def _bm25_similarity(self, sentence, entity_idx):
        tok_sentence = jieba.lcut(sentence)
        return self.bm25_model.get_score(tok_sentence, entity_idx)

    def _get_e_id(self, entity):
        return self.full_entity_list.index(entity)

    def _calc_features(self, mention, entity, question, mask_idx=[]):
        string_sim = self._string_similarity(mention, entity)
        m_e_w2v_sim = self._word2vec_similarity(mention, entity)
        # m_s_w2v_sim = self._word2vec_similarity(mention, question)
        e_pop = self._entity_popularity(entity)
        # m_e_bm25_sim = self._bm25_similarity(mention, self._get_e_id(entity))
        m_s_bm25_sim = self._bm25_similarity(question, self._get_e_id(entity))
        reg_lev_dist = self._levenshstein_distance(mention, entity)
        all_features = [string_sim, m_e_w2v_sim, e_pop, m_s_bm25_sim, reg_lev_dist]
        for idx in mask_idx:
            all_features.pop(idx)
        return all_features

    def _parse_dataset(self, path, get_group=False):
        features, labels, group_sizes = [], [], []
        print(f"Parsing {path}...")
        
        file_name = osp.basename(path)
        dir_name = osp.dirname(path)
        cached_path = osp.join(dir_name, f"cached_{file_name}")
        if osp.exists(cached_path):
            try:
                print(f"Loading cached {cached_path}")
                return pickle.load(open(cached_path, 'rb'))
            except:
                print(f"Loading {cached_path} failed, regenerating...")
    
        with open(path, 'r') as f:
            cur_mention, cur_original_q = "", ""
            for line in tqdm(f.readlines()):
                if line == '\n':
                    continue
                mention, entity, question, label, original_q = line.split('\t')
                if not entity:
                    # invalid data
                    continue
                
                if mention != cur_mention or original_q != cur_original_q:
                    cur_mention = mention
                    cur_original_q = original_q
                    group_sizes.append(1)
                else:
                    group_sizes[-1] += 1
                
                features.append(self._calc_features(mention, entity, question))
                labels.append(int(label))
        if get_group:
            pickle.dump((np.array(features), np.array(labels), np.array(group_sizes)), open(cached_path, 'wb'))
            return np.array(features), np.array(labels), np.array(group_sizes)
        else:
            pickle.dump((np.array(features), np.array(labels)), open(cached_path, 'wb'))
            return np.array(features), np.array(labels)

    def load_model(self, path):
        self.el_model = pickle.load(open(path, 'rb'))

    def _do_test(self, test_features, group):
        now_idx = 0
        test_preds = []
        for group_size in tqdm(group):
            test_pred = self.el_model.predict(test_features[now_idx:now_idx + group_size])
            test_preds.append(np.argmax(test_pred))
            now_idx += group_size
        return sum([1 if pred == 0 else 0 for pred in test_preds]) / len(test_preds)

    def train(self, train_args):
        train_features, train_labels, group = self._parse_dataset(train_args.train_file, get_group=True)
        dev_features, dev_labels, dev_group = self._parse_dataset(train_args.dev_file, get_group=True)
        self.el_model.fit(train_features, train_labels, group=group)
        pickle.dump(self.el_model, open(os.path.join(train_args.output_dir, 'best_el.bin'), 'wb'))
        acc = self._do_test(dev_features, dev_group)
        print(f"Eval accuracy: {acc}")

    def test(self, test_args):
        test_features, test_labels, group = self._parse_dataset(test_args.test_file, get_group=True)
        acc = self._do_test(test_features, group)
        print(f"Test accuracy: {acc}")
        # print(f"Test F1 Score: {f1}")
    
    def ablation_test(self, train_args, test_args):
        num_features = 7
        for idx in range(num_features):
            train_features, train_labels, train_group = self._parse_dataset(train_args.train_file, get_group=True)
            test_features, test_labels, test_group = self._parse_dataset(test_args.test_file, get_group=True)
            train_features = np.delete(train_features, idx, axis=1)
            # train_labels = np.delete(train_labels, idx, axis=1)
            test_features = np.delete(test_features, idx, axis=1)
            # test_labels = np.delete(test_labels, idx, axis=1)
            self.el_model.fit(train_features, train_labels, group=train_group)
            acc = self._do_test(test_features, group=test_group)
            print(f"Test accuracy: {acc}")

    def _top_k_bm25(self, question, candidate_entities, top_k=10):
        question = jieba.lcut(question)
        bm25_dists = [self.bm25_model.get_score(question, self._get_e_id(e)) for e in candidate_entities]
        if np.allclose(bm25_dists, [0] * len(bm25_dists)):
            return candidate_entities
        topk_idx = np.argpartition(bm25_dists, -top_k)[-top_k:]           
        return [candidate_entities[idx] for idx in topk_idx]

    def _top_k_w2v(self, mention, candidate_entities, top_k=10):
        w2v_dists = [self._word2vec_similarity(mention, e) for e in candidate_entities]
        topk_idx = np.argpartition(w2v_dists, -top_k)[-top_k:]           
        return [candidate_entities[idx] for idx in topk_idx]


    def get_entity(self, mentions, question, candidate_entities, el_threshold=0, top_k=1, best_entitiy_only=True, pre_top_k=10):
        results = []
        for mention in mentions:
            if mention in candidate_entities:
                # perfect matching
                results.append([[mention, 1]])
            else:
                candidate_entities = self._top_k_w2v(mention, candidate_entities, pre_top_k)
                m_results, features = [], []
                for entity in candidate_entities:
                    features.append(self._calc_features(mention, entity, question))
                m_results = self.el_model.predict(features)
                entity_prob_list = [[e, r] for e, r in zip(candidate_entities, m_results) if r > el_threshold]
                results.append(sorted(entity_prob_list, key=lambda x: x[1], reverse=True)[:top_k])
        if best_entitiy_only:
            return [] if any(len(r) == 0 for r in results) else [r[0][0] for r in results]
        return results
            
