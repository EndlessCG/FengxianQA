QUESTION_INTENTS = {
    # (e)-(a) ret a ex.仓库准入的风险等级是什么？
    # 'e'=entity, 'a'=attribute, 'r'=relation, 'v'=value
    'one_hop_a':{
        'answer': "{}的{}是{}",
        'answer_slots': [('e', 0), ('a', 0), ('v', 0)],
        'query': "match (n) where n.`名称` = '{}' return n.`{}`",
        'query_slots': [('e', 0), ('a', 0)],
        'n_hops_': 1,
        'get_all_query_': "match (e) return e",
        'question_':['{}的{}是什么？']
    },

    # (e)-[r]-(e1) ret e1.name ex.业务准备包含哪些环节？
    'one_hop_e':{
        'answer': "{}{}{}",
        'answer_slots': [('e', 0), ('r', 0), ('v', 0)],
        'query': "match (n)-[r]->(m) where n.`名称` = '{}' and type(r)= '{}' return m.`名称`",
        'query_slots': [('e', 0), ('r', 0)],
        'n_hops_': 1,
        'get_all_query_': "match (e)-[r]->(e1) return e, r, e1`",
    },

    # (e)-(e1)-(a) ret a ex.业务准备的各环节由哪些部门负责？
    'two_hop_a':{
        'answer': "{}{}的{}的{}是{}",
        'answer_slots': [('e', 0), ('r', 0), ('pron', 0), ('a', 0), ('v', 0)],
        'n_hops': 2,
        'cypher': "(e)-[r]->(e1)",
        'question': "{}{}的{}的{}是什么",
        'raw_answer_slots': ["values_"],
        'question_slots': ["e.`名称`", "type(r)", "labels(e1)", "properties(e1)"],
        'triples': [["e.`名称`", "type(r)", -1], [-1, "keys_", 'values_']]
    },

    # (e)-(e1)-(e2) ret e2 ex.业务准备的各环节有哪些风险点？
    'two_hop_e':{
        'answer': '{}{}的{}{}的{}是{}',
        'answer_slots':[('e', 0), ('r', 0), ('pron', 0), ('r', 1), ('pron', 1), ('v', 0)],
        'n_hops': 2,
        'cypher': "(e)-[r]->(e1)-[r1]->(e2)",
        'question': "{}{}的{}{}的{}有哪些",
        'raw_answer_slots': ["e2.`名称`"],
        'question_slots': ["e.`名称`", "type(r)", "labels(e1)", "type(r1)", "labels(e2)"],
        'triples': [["e.`名称`", "type(r)", -1], [-1, "type(r1)", "e2.`名称`"]]
    },


    # 暂不支持
    # (e)-(e1)-(a) ret e1 ex.业务准备中由运营中心负责的有哪些环节？
    'two_hop_e_with_a':{
        'cypher': "(e)-[r]->(e1)",
    },

    # (e)-(e1)-(e2) ret e1 ex.业务准备中有合规性风险的是哪个环节？
    'two_hop_e_with_e':{
        'cypher': "(e)-[r]->(e1)-[r1]->(e2)"
    },
    
    'one_hop_e_with_a':{
        'cypher': "(e)",
    },

    'one_hop_e_with_e':{
        'cypher': "(e)-[r]->(e1)",
    }
}