from utils import load_el_questions, Neo4jGraph
from config import neo4j_config
from models.EL.EL_model import EL
from models.EL.EL_main import parse_args
from tqdm import tqdm
el_questions = load_el_questions('input/data/el/test.txt')

def main():
    graph = Neo4jGraph(neo4j_config['neo4j_addr'], \
                        neo4j_config['username'], \
                        neo4j_config['password'])
    all_entities = graph.entity_list + graph.attribute_list
    train_args = parse_args()
    model = EL(train_args, graph)
    
    for strategy in ['q_e_bm25', 'm_e_bm25', 'q_e_w2v', 'm_e_w2v']:
        for top_k in [1, 3, 5, 10]:
            catch_entity_cnt = 0
            total_cnt = 0
            for v in tqdm(el_questions.values()):
                for mention, question, entity in v:
                    total_cnt += 1
                    if entity in model._filter_entities(mention, question, all_entities, top_k, strategy):  
                        catch_entity_cnt += 1 
            print(f"{strategy} top {top_k} recall {catch_entity_cnt / total_cnt}")
            

if __name__ == '__main__':
    main()