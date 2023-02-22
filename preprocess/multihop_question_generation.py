import pandas as pd
from utils.question_intents import QUESTION_INTENTS
from utils.neo4j_graph import Neo4jGraph

graph = Neo4jGraph("bolt://localhost:7687", 'neo4j', '123456')

def get_mhop_data(intent_list):
    question_list, triple_list, answer_list = [], [], []
    for intent in intent_list:
        sketch = QUESTION_INTENTS[intent]['cypher']
        slots = QUESTION_INTENTS[intent]['question_slots']
        for raw_answer_slot in QUESTION_INTENTS[intent]['raw_answer_slots']:
            if raw_answer_slot != 'values_' and raw_answer_slot != 'keys_':
                slots.append(raw_answer_slot)
        query = f"match {sketch} return distinct {','.join(slots)}"
        components_list = graph.execute_query(query, drop_keys=False)
        temp_list = []
        for component in components_list:
            temp_component = {}
            for key, value in component.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        temp_component.setdefault('keys_', []).append(k)
                        temp_component.setdefault('values_', []).append(v)
                else:
                    temp_component[key] = value
            temp_list.append(temp_component)
        components_list = pd.DataFrame(temp_list)
        # 展开列表
        columns_list = list(components_list.columns)
        for column in columns_list:
            if column == 'keys_' or column == 'values_':
                continue
            components_list = components_list.explode(column)
        if 'keys_' in columns_list:
            components_list = components_list.explode(column=['keys_', 'values_'])
        
        for _, component in components_list.iterrows():
            slot_fills = []
            for key in QUESTION_INTENTS[intent]['question_slots']:
                if 'properties' in key:
                    true_key = 'keys_'
                else:
                    true_key = key
                slot_fills.append(component[true_key])

            question = [QUESTION_INTENTS[intent]['question'].format(*slot_fills)]
            answer_slots = QUESTION_INTENTS[intent]['raw_answer_slots']
            answer = component[answer_slots[0]]
            true_triples = []
            for triple in QUESTION_INTENTS[intent]['triples']:
                true_triple = []
                for item in triple:
                    if item == -1:
                        true_triple.append('?x')
                    else:
                        true_triple.append(component[item])
                true_triples.append(true_triple)
                
            question_list += question
            answer_list += [answer]
            triple_list += [true_triples]
    return question_list, triple_list, answer_list

def write_data_file(path, question_list, triple_list, answer_list, start_id=1294):
    with open(path, 'w+') as f:
        for i, (q, t, a) in enumerate(zip(question_list, triple_list, answer_list)):
            idx = start_id + i
            f.write(f"<question id={idx}> {q}\n")
            for ti, tr in enumerate(t):
                f.write(f"<triple{ti} id={idx}> {tr[0]}\t{tr[1]}\t{tr[2]}\n")
            f.write(f"<answer id={idx}> {a}\n")

def main():
    question_list, triple_list, answer_list = get_mhop_data(['two_hop_a', 'two_hop_e'])
    write_data_file("models/input/data/fengxian/qa/QA_mhop_data.txt", question_list, triple_list, answer_list)


if __name__ == '__main__':
    main()
