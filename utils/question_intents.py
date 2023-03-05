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

    'TaA': {
        'answer': ["{}为{}的是{}"],
        'answer_slots': [[('l', 0), ('a', 0), ('v', 0)]],
        'display': '属性约束求实体',
        'query': "match (n) \
                  where n.`{}`='{}' \
                  return distinct n.`名称`",
        'query_slots': [('l', 0), ('a', 0)],
    },

    'TeE': {
        'answer': ["{}{}的是{}"],
        'answer_slots': [[('l', 0), ('e', 0), ('v', 0)]],
        'display': '实体约束求实体',
        'query': "match (n)-[r]->(n1) \
                  where n1.`名称`='{}' and type(r)='{}' \
                  return n.`名称`",
        'query_slots': [('e', 0), ('l', 0)],
    },

    # (e)-(e1)-(a) ret a ex.业务准备的各环节由哪些部门负责？
    'EeNaT': {
        'answer': ["{}{}{}", "{}的{}是{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)], [('v', 0), ('l', 1), ('v', 1)]],
        'display': '两跳求属性',
        'query': "match (n)-[r]->(n1) where n.`名称` = '{}' and type(r)='{}' return n1.`名称`, n1.`{}`",
        'query_slots': [('e', 0), ('l', 0), ('l', 1)],
    },

    # (e)-(e1)-(e2) ret e2 ex.业务准备的各环节有哪些风险点？
    'EeNeT': {
        'answer': ["{}{}{}", "{}{}{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)], [('v', 0), ('l', 1), ('v', 1)]],
        'display': '两跳求实体',
        'query': "match (n)-[r]->(n1)-[r1]->(n2) where n.`名称` = '{}' and type(r)='{}' and type(r1)='{}' return n1.`名称`, n2.`名称`",
        'query_slots': [('e', 0), ('l', 0), ('l', 1)],
    },

    # (e)-(e1)-(a) ret e1 ex.业务准备中由运营中心负责的有哪些环节？
    'EeTaA': {
        'answer': ["{}{}{}", "{}为{}的是{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)], [('l', 1), ('a', 0), ('v', 1)]],
        'display': '单跳+属性约束求实体',
        'query': "match (n)-[r]->(n1) \
                  where n.`名称`='{}' and n1.`{}`='{}'\
                  return distinct n1.`名称`",
        'query_slots': [('e', 0), ('l', 0), ('a', 0)],
    },

    # (e)-(e1)-(e2) ret e1 ex.业务准备中有合规性风险的是哪个环节？
    'EeTeE': {
        'answer': ["{}{}{}", "{}{}的是{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)], [('l', 1), ('e', 1), ('v', 1)]],
        'display': '单跳+实体约束求实体',
        'query': "match (n)-[r]->(n1)-[r1]->(n2) \
              where n.`名称`='{}' and n2.`名称`='{}' \
              return distinct n1.`名称`",
        'query_slots': [('e', 0), ('e', 1)],
    },

    'TeNaA': {
        'answer': ["{}{}为{}的是{}"],
        'answer_slots': [[('l', 0), ('l', 1), ('a', 0), ('v', 0)]],
        'display': '含属性约束的实体约束求实体',
        'query': "match (n)-[r]->(n1) \
              where type(r)='{}' and n1.`{}`='{}' \
              return distinct n.`名称`",
        'query_slots': [('l', 0), ('l', 1), ('a', 0)],
    },

    'TeNeE': {
        'answer': ["{}{}{}的是{}"],
        'answer_slots': [[('l', 0), ('l', 1), ('e', 0), ('v', 0)]],
        'display': '含实体约束的实体约束求实体',
        'query': "match (n)-[r]->(n1)-[r1]->(n2) \
              where type(r)='{}' and type(r1)='{}' and n2.`名称`='{}' \
              return distinct n1.`名称`",
        'query_slots': [('l', 0), ('l', 1), ('e', 0)],
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
    'TaA': "match (n) \
            unwind [k in keys(n) where n[k]='{attribute}'] as k \
            return distinct '[TARGET]'+k",
    'TeE': "match (n)-[r]->(n1) \
            where n1.`名称`='{entity}' \
            return distinct '[TARGET]'+type(r)",
    'EeNaT': "match (n)-[r]->(n1) \
            where n.`名称`='{entity}' \
            unwind keys(n1) as attr \
            return distinct type(r)+'[NEDGE]'+attr+'[TARGET]'",
    'EeNeT': "match (n)-[r]->()-[r1]->() \
              where n.`名称`='{entity}' \
              return distinct type(r)+'[NEDGE]'+type(r1)+'[TARGET]'",
    'EeTaA': "match (n)-[r]->(n1) \
              where n.`名称`='{entity}' \
              unwind [k in keys(n) where n[k]='{attribute}'] as k \
              return distinct type(r)+'[TARGET]'+k",
    'EeTeE': "match (n)-[r]->(n1)-[r1]->(n2) \
              where n.`名称`='{entity}' and n2.`名称`='{entity1}' \
              return distinct type(r)+'[TARGET]'+type(r1)",
    'TeNaA': "match ()-[r]->(n) \
              unwind [k in keys(n) where n[k]='{attribute}'] as k \
              return distinct '[TARGET]'+type(r)+'[NEDGE]'+k",
    'TeNeE': "match ()-[r]->(n1)-[r1]->(n2) \
              where n2.`名称`='{entity}' \
              return distinct '[TARGET]'+type(r)+'[NEDGE]'+type(r1)"
}
