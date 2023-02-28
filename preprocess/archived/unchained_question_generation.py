import pandas as pd
import numpy as np
# 处理风险点
fengxiandian_path = "preprocess/风控实体关系表/entity_fengxiandian.csv"
output_path = "input/data/fengxian/qa/QA_unchained_mhop_data.txt"
fengxiandian = pd.read_csv(fengxiandian_path)
# 处理专有名词
mingci_path = "preprocess/风控实体关系表/entity_mingci.csv"
mingci = pd.read_csv(mingci_path)
# 处理业务流程
yewuliucheng_path = "preprocess/风控实体关系表/entity_yewuliucheng.csv"
yewuliucheng = pd.read_csv(mingci_path)
# 处理业务环节
yewuhuanjie_path = "preprocess/风控实体关系表/entity_yewuhuanjie.csv"
yewuhuanjie = pd.read_csv(yewuhuanjie_path, keep_default_na=False)
# 处理包含
baohan_path = "preprocess/风控实体关系表/relationship_baohan.csv"
baohan = pd.read_csv(baohan_path, keep_default_na=False)
# 处理预警
yujing_path = "preprocess/风控实体关系表/entity_yujing.csv"
yujing = pd.read_csv(yujing_path, keep_default_na=False)

question_desc_list = [
    # one_hop_e_with_a
    [
        # 风险点-风险等级
        fengxiandian, # charts
        ["风险等级为{}的风险点有哪些？", "风险等级为{}的风险点有什么？", "{}级别的风险点有哪些？"], # questions
        ['风险等级'], # question slot fills
        [[':ID', '=风险等级', '风险等级']], # triples
        ':ID', # answer
        [('风险等级', 'attribute')], # ner
        '<REDGE>{}', # path
        ['风险等级'] # path slot sills
    ],
    [
        # 名词-类型
        mingci,
        ["{}名词有哪些？", "哪些名词属于{}？"],
        ['类型'],
        [[':ID', '=类型', '类型']],
        ':ID',
        [('类型', 'attribute')], # ner
        '<REDGE>{}', # path
        ['类型'] # path slot sills
    ],
    [
        # 预警-类型
        yujing,
        ["{}预警有哪些？", "哪些预警属于{}？"],
        ['类型'],
        [[':ID', '=类型', '类型']],
        ':ID',
        [('类型', 'attribute')], # ner
        '<REDGE>{}', # path
        ['类型'] # path slot sills
    ],
    [
        # 业务环节-负责角色
        yewuhuanjie,
        ["{}部门负责的有哪些业务？", "{}负责什么业务？"],
        ['负责角色'],
        [[':ID', '=负责角色', '负责角色']],
        ':ID',
        [('负责角色', 'attribute')], # ner
        '<REDGE>{}', # path
        ['负责角色'] # path slot sills
    ],
    # [
    #     # x-含义，预警-x色预警
    # ],

    # one_hop_e_with_e
    [
        # 业务环节包含风险
        pd.merge(baohan, yewuhuanjie, left_on=':START_ID', right_on=':ID', how='inner'),
        ["包含{}的业务环节是什么？", "包含{}的业务环节有哪些？", "什么业务有{}？", "哪些业务有{}？"],
        [':END_ID'],
        [[':START_ID', '=包含', ':END_ID']],
        ':START_ID',
        [(':END_ID', 'entity')], # ner
        '<REDGE>{}', # path
        [':END_ID'] # path slot sills
    ],
    [
        # 业务流程包含业务环节
        pd.merge(baohan, yewuliucheng, left_on=':START_ID', right_on=':ID', how='inner'),
        ["包含{}的业务环节是什么？", "包含{}的业务环节有哪些？", "什么业务有{}？", "哪些业务有{}？"],
        [':END_ID'],
        [[':START_ID', '=包含', ':END_ID']],
        ':START_ID',
        [(':END_ID', 'entity')], # ner
        '<REDGE>{}', # path
        [':END_ID'] # path slot sills
    ],

    # two_hop_e_with_a e-[e]-a
    [
        # 业务环节-包含-[风险]-风险等级
        pd.merge(
            pd.merge(yewuhuanjie, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
            fengxiandian, left_on=':END_ID', right_on=':ID', how='inner'
        ),
        ["{}业务包含的{}等级风险有哪些？"],
        [':START_ID', '风险等级'],
        [[':START_ID', '=包含', ':END_ID'], [':END_ID', '=风险等级', '风险等级']],
        ':END_ID',
        [(':START_ID', 'entity'), ('风险等级', 'attribute')], # ner
        '{}<NEDGE>{}<REDGE>{}', # path
        [':END_ID'] # path slot sills
    ]
]

start_id = 1765
with open(output_path, 'w+') as f:
    for chart, raw_questions, question_slots, raw_triples, answer_idx in question_desc_list:
        questions = sum([[raw_question.format(*val) for val in chart[question_slots].values] for raw_question in raw_questions], [])
        tripless = []
        for raw_triple in raw_triples:
            triples = []
            explode_idx = []
            for i, title in enumerate(raw_triple):
                if title[0] == '=':
                    triples.append([title[1:]])
                else:
                    triples.append(chart[title].values.squeeze().tolist())
            triples = pd.DataFrame(triples).T.fillna(method='pad').values.tolist()
            tripless.append(triples)
        tripless = [[tripless[i][j] for i in range(len(tripless))] for j in range(len(tripless[0]))]
        answers = chart[answer_idx]
        for i, (question, triples, answer) in enumerate(zip(questions, tripless, answers)):
            id = i + start_id
            f.write(f"<question id={id}> {question}\n")
            for j, triple in enumerate(triples):
                f.write("<triple{} id={}> {}\n".format(j if len(triples) > 1 else '', id, '\t'.join(triple)))
            
            f.write(f"<answer id={id}> {answer}\n")
