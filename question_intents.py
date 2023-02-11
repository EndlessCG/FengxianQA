QUESTION_INTENTS = {
    # (e)-(a) ret a ex.仓库准入的风险等级是什么？
    # 'e'=entity, 'a'=attribute, 'r'=relation, 'v'=value
    'one_hop_a':{
        'answer': "{}的{}是{}",
        'answer_slots': [('e', 0), ('a', 0), ('v', 0)],
        'query': "match (n) where n.`名称` = '{}' return n.`{}`",
        'query_slots': [('e', 0), ('a', 0)],
    },

    # (e)-[r]-(e1) ret e1.name ex.业务准备包含哪些环节？
    'one_hop_e':{
        'answer': "{}{}{}",
        'answer_slots': [('e', 0), ('r', 0), ('v', 0)],
        'query': "match (n)-[r]->(m) where n.`名称` = '{}' and type(r)= '{}' return m.`名称`",
        'query_slots': [('e', 0), ('r', 0)]
    },

    # (e)-(e1)-(a) ret a ex.业务准备的各环节由哪些部门负责？
    'two_hop_a':{
        
    },

    # (e)-(e1)-(e2) ret e2 ex.业务准备的各环节有哪些风险点？
    'two_hop_e':{

    },

    # (e)-(e1)-(a) ret e1 ex.业务准备中由运营中心负责的有哪些环节？
    'two_hop_e_with_a':{

    },

    # (e)-(e1)-(e2) ret e1 ex.业务准备中有合规性风险的是哪个环节？
    'two_hop_e_with_e':{

    }
}