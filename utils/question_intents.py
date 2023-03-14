QUESTION_INTENTS = {
    # (e)-(a) ret a ex.仓库准入的风险等级是什么？
    # 'e'=entity, 'l'=link, 'v'=value, 'a'=attribute
    'EaT': {
        'answer': ["{}的{}是{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)]],
        'display': '单跳求属性',
        'query': "match (n) where n.`名称` = '{}' return n.`{}`",
        'query_slots': [('e', 0), ('l', 0)],
    },

    # (e)-[r]-(e1) ret e1.name ex.业务准备包含哪些环节？
    'EeT': {
        'answer': ["{}{}{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)]],
        'display': '单跳求实体',
        'query': "match (n)-[r]->(m) where n.`名称` = '{}' and type(r)= '{}' return m.`名称`",
        'query_slots': [('e', 0), ('l', 0)],
    },
}

SUBGRAPHS = {
    'EaT': "match (n) \
            where n.`名称`='{entity}' \
            unwind keys(n) as attr \
            return distinct attr+'[TARGET]'",
    'EeT': "match (n)-[r]->() \
            where n.`名称`='{entity}' \
            return distinct type(r)+'[TARGET]'",
}
