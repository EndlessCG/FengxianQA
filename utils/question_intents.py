QUESTION_INTENTS = {
    # (e)-(a) ret a ex.仓库准入的风险等级是什么？
    # 'e'=entity, 'l'=link, 'v'=value
    # slots中e个数=get_sgraph_query_中entity个数，
    # a个数=get_sgraph_query_中attribute个数，
    # l个数=len(sgraph.split('[TARGET], '[NEDGE]'))，
    # v个数=query返回值个数
    'EaT': {
        'answer': ["{}的{}是{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)]],
        'display': '单跳求属性',
        'query': "match (n) where n.`名称` = '{}' return n.`{}`",
        'query_slots': [('e', 0), ('l', 0)],
        'n_hops': 1,
        'get_all_query_': "match (e) return e",
        'get_sgraph_query_': "match (n)-[r]->() \
                            where n.`名称`='{entity}' \
                            return distinct type(r)+'[TARGET]'"
    },

    # (e)-[r]-(e1) ret e1.name ex.业务准备包含哪些环节？
    'EeT': {
        'answer': ["{}{}{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)]],
        'display': '单跳求实体',
        'query': "match (n)-[r]->(m) where n.`名称` = '{}' and type(r)= '{}' return m.`名称`",
        'query_slots': [('e', 0), ('l', 0)],
        'n_hops': 1,
        'get_all_query_': "match (e)-[r]->(e1) return e, r, e1`",
        'get_sgraph_query_': "match (n) \
            where n.`名称`='{entity}' \
            return distinct keys(n)+'[TARGET]'"
    },

    'TaA': {
        'answer': ["{key}为{attribute}的是{result}"],
        'query': "",
        'n_hops_': 1,
        'get_sgraph_query': "match (n) \
                            with filter(key in keys(n) where n[key]='{attribute}') as k \
                            where not size(k)=0 \
                            return distinct '[TARGET]'+k"
    },

    'TeE': {
        'get_sgraph_query': "match (n)-[r]->(n1) \
                            where n1.`名称`='{entity}' \
                            return distinct '[TARGET]'+type(r)"
    },

    # (e)-(e1)-(a) ret a ex.业务准备的各环节由哪些部门负责？
    'EeNaT': {
        'answer': ["{}{}{}", "{}的{}是{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)], [('v', 0), ('l', 1), ('v', 1)]],
        'display': '两跳求属性',
        'query': "match (n)-[r]->(n1) where n.`名称` = '{}' and type(r)='{}' return n1.`名称`, n1.`{}`",
        'query_slots': [('e', 0), ('l', 0), ('l', 1)],
        'n_hops': 2,
        'get_sgraph_query': "match (n)-[r]->(n1) \
            where n.`名称`='{entity}' \
            unwind keys(n1) as attr \
            return distinct type(r)+'[NEDGE]'+attr+'[TARGET]'",
    },

    # (e)-(e1)-(e2) ret e2 ex.业务准备的各环节有哪些风险点？
    'EeNeT': {
        'answer': ["{}的{}是{}", "{}的{}是{}"],
        'answer_slots': [[('e', 0), ('l', 0), ('v', 0)], [('v', 0), ('l', 1), ('v', 1)]],
        'display': '两跳求实体',
        'query': "match (n)-[r]->(n1)-[r1]->(n2) where n.`名称` = '{}' and type(r)='{}' and type(r1)='{}' return n1.`名称`, n2.`名称`",
        'query_slots': [('e', 0), ('l', 0), ('l', 1)],
        'n_hops': 2,
        'get_sgraph_query': "match (n)-[r]->()-[r1]->() \
              where n.`名称`='{entity}' \
              return distinct type(r)+'[NEDGE]'+type(r1)+'[TARGET]'",
    },

    # (e)-(e1)-(a) ret e1 ex.业务准备中由运营中心负责的有哪些环节？
    'EeTaA': {
        'get_sgraph_query': "match (n)-[r]->(n1) \
              where n.`名称`='{entity}' \
              with filter(key in keys(n1) where n1[key]='{attribute}') as k \
              return distinct type(r)+'[TARGET]'+k",
    },

    # (e)-(e1)-(e2) ret e1 ex.业务准备中有合规性风险的是哪个环节？
    'EeTeE': {
        'get_sgraph_query': "match (n)-[r]->(n1)<-[r1]-(n2) \
              where n.`名称`={entity} and n2.`名称`={entity1} \
              return distinct type(r)+'[TARGET]'+type(r1)'",
    },

    'TeNaA': {
        'get_sgraph_query': "match ()-[r]->(n1) \
              with filter(key in keys(n1) where n1[key]='{attribute}') as k \
              return distinct '[TARGET]'+type(r)+'[NEDGE]'+a",
    },

    'TeNeE': {
        'get_sgraph_query': "match ()-[r]->(n1)-[r1]->(n2) \
              where n2.`名称`={entity} \
              return distinct '[TARGET]'+type(r)+'[NEDGE]'+type(r1)"
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
            with filter(key in keys(n) where n[key]='{attribute}') as k \
            where not size(k)=0 \
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
              with filter(key in keys(n1) where n1[key]='{attribute}') as k \
              return distinct type(r)+'[TARGET]'+k",
    'EeTeE': "match (n)-[r]->(n1)<-[r1]-(n2) \
              where n.`名称`='{entity}' and n2.`名称`='{entity1}' \
              return distinct type(r)+'[TARGET]'+type(r1)",
    'TeNaA': "match ()-[r]->(n1) \
              with filter(key in keys(n1) where n1[key]='{attribute}') as k \
              return distinct '[TARGET]'+type(r)+'[NEDGE]'+a",
    'TeNeE': "match ()-[r]->(n1)-[r1]->(n2) \
              where n2.`名称`='{entity}' \
              return distinct '[TARGET]'+type(r)+'[NEDGE]'+type(r1)"
}
