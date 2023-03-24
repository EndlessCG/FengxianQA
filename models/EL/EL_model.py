# dataset: https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d100-v0.2.0-s.tar.gz
from gensim.models import KeyedVectors
from gensim.summarization.bm25 import BM25
import jieba
import pickle
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

class EL():
    def __init__(self, args, neo4j_graph):
        self.args = args
        print("Loading w2v model...")
        if hasattr(args, "w2v_load_path") and osp.exists(args.w2v_load_path):
            self.w2v_model = pickle.load(open(args.w2v_load_path, 'rb'))
        else:
            self.w2v_model = KeyedVectors.load_word2vec_format(args.w2v_corpus_path, binary=False)
            if not os.path.exists(osp.dirname(args.w2v_save_path)):
                os.makedirs(osp.dirname(args.w2v_save_path))
            pickle.dump(self.w2v_model, open(args.w2v_save_path, 'wb'), protocol=4)
        self.graph = neo4j_graph
        self.full_entity_list = self.graph.entity_list + self.graph.attribute_list
        print("Tokenizing entity list...")
        tok_entity_list = [jieba.lcut(entity) for entity in self.full_entity_list]
        print("Building BM25...")
        self.bm25_model = BM25(tok_entity_list)
        xgb.set_config(verbosity=2)
        self.el_model = XGBClassifier()
        print('w2v', type(self.w2v_model))
    
    def _string_similarity(self, mention, entity):
        # Jaccard similarity
        tok_mention = jieba.lcut(mention)
        tok_entity = jieba.lcut(entity)
        intersection = set(tok_mention) & set(tok_entity)
        union = set(tok_mention) | set(tok_entity)
        return len(intersection) / len(union)
    
    def _word2vec_similarity(self, s1, s2):
        cut_s1, cleaned_s1 = jieba.lcut(s1), []
        cut_s2, cleaned_s2 = jieba.lcut(s2), []
        for w1 in cut_s1:
            if w1 not in self.w2v_model:
                cleaned_s1.extend(list(w1))
            else:
                cleaned_s1.append(w1)
        for w2 in cut_s2:
            if w2 not in self.w2v_model:
                cleaned_s2.extend(list(w2))
            else:
                cleaned_s2.append(w2)
        if len(cleaned_s1) == 0 or len(cleaned_s2) == 0:
            return 0
        return self.w2v_model.n_similarity(cleaned_s1, cleaned_s2)

    def _entity_popularity(self, entity):
        pop_query = f"match (n) where n.`名称`='{entity}' with size((n)-[]-()) as degree return degree;"
        result = self.graph.execute_query(pop_query)
        return 0 if result == [] else result[0]

    def _bm25_similarity(self, sentence, entity_idx):
        tok_sentence = jieba.lcut(sentence)
        return self.bm25_model.get_score(tok_sentence, entity_idx)

    def _get_e_id(self, entity):
        return self.full_entity_list.index(entity)

    def _calc_features(self, mention, entity, question):
        string_sim = self._string_similarity(mention, entity)
        m_e_w2v_sim = self._word2vec_similarity(mention, entity)
        m_s_w2v_sim = self._word2vec_similarity(mention, question)
        e_pop = self._entity_popularity(entity)
        bm25_sim = self._bm25_similarity(question, self._get_e_id(entity))
        return [string_sim, m_e_w2v_sim, m_s_w2v_sim, e_pop, bm25_sim]

    def _parse_dataset(self, path):
        features, labels = [], []
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
            for line in tqdm(f.readlines()):
                mention, entity, question, label = line.split('\t')
                if not entity:
                    # invalid data
                    continue
                features.append(self._calc_features(mention, entity, question))
                labels.append(int(label))
        pickle.dump((np.array(features), np.array(labels)), open(cached_path, 'wb'))
        return np.array(features), np.array(labels)

    def load_model(self, path):
        self.el_model = pickle.load(open(path, 'rb'))

    def eval(self, y_true, y_pred):
        assert len(y_true) == len(y_pred), "True & pred length mismatch"
        eval_f1 = f1_score(y_true, y_pred)
        eval_acc = sum([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))]) / len(y_true)
        return eval_acc, eval_f1

    def train(self, train_args):
        train_features, train_labels = self._parse_dataset(train_args.train_file)
        dev_features, dev_labels = self._parse_dataset(train_args.dev_file)
        self.el_model.fit(train_features, train_labels)
        pickle.dump(self.el_model, open(os.path.join(train_args.output_dir, 'best_el.bin'), 'wb'))
        dev_pred = self.el_model.predict(dev_features)
        acc, f1 = self.eval(dev_labels, dev_pred)
        print(f"Eval accuracy: {acc}")
        print(f"Eval F1 Score: {f1}")

    def test(self, test_args):
        test_features, test_labels = self._parse_dataset(test_args.test_file)
        test_pred = self.el_model.predict(test_features)
        acc, f1 = self.eval(test_labels, test_pred)
        print(f"Test accuracy: {acc}")
        print(f"Test F1 Score: {f1}")
    
    def get_entity(self, mentions, question, top_k=1):
        results = []
        for mention in mentions:
            m_results = []
            for entity in self.full_entity_list:
                features = self._calc_features(mention, entity, question)
                m_results.append([entity, self.el_model.predict(features)])
            results.append()
            
            
        
