import pandas as pd

CSV_PATHS = {
    # entity
    "fengxiandian": "preprocess/风控实体关系表/entity_fengxiandian.csv",
    "mingci": "preprocess/风控实体关系表/entity_mingci.csv",
    "yewuliucheng": "preprocess/风控实体关系表/entity_yewuliucheng.csv",
    "yewuhuanjie": "preprocess/风控实体关系表/entity_yewuhuanjie.csv",
    "yujing": "preprocess/风控实体关系表/entity_yujing.csv",
    # relation
    "baohan": "preprocess/风控实体关系表/relationship_baohan.csv",
}

def get_question_descriptions():
    # 处理风险点
    fengxiandian_path = "preprocess/风控实体关系表/entity_fengxiandian.csv"
    fengxiandian = pd.read_csv(fengxiandian_path)
    # 处理专有名词
    mingci_path = "preprocess/风控实体关系表/entity_mingci.csv"
    mingci = pd.read_csv(mingci_path)
    # 处理业务流程
    yewuliucheng_path = "preprocess/风控实体关系表/entity_yewuliucheng.csv"
    yewuliucheng = pd.read_csv(yewuliucheng_path)
    # 处理业务环节
    yewuhuanjie_path = "preprocess/风控实体关系表/entity_yewuhuanjie.csv"
    yewuhuanjie = pd.read_csv(yewuhuanjie_path, keep_default_na=False)
    # 处理包含
    baohan_path = "preprocess/风控实体关系表/relationship_baohan.csv"
    baohan = pd.read_csv(baohan_path, keep_default_na=False)
    # 处理预警
    yujing_path = "preprocess/风控实体关系表/entity_yujing.csv"
    yujing = pd.read_csv(yujing_path, keep_default_na=False)

    return [
        # two_hop_e
        [
            
        ]
        # one_hop_e_with_a
        [
            # 风险点-风险等级
            fengxiandian, # charts
            ["风险等级为{}的风险点有哪些？", "风险等级为{}的风险点有什么？", "{}级别的风险点有哪些？"], # questions
            ['风险等级'], # question slot fills
            [('风险等级', 'attribute')], # ner
            '[TARGET]风险等级', # path
        ],
        [
            # 名词-类型
            mingci,
            ["{}名词有哪些？", "哪些名词属于{}？"],
            ['类型'],
            [('类型', 'attribute')], # ner
            '[TARGET]类型', # path
        ],
        [
            # 预警-类型
            yujing,
            ["{}预警有哪些？", "哪些预警属于{}？"],
            ['类型'],
            [('类型', 'attribute')], # ner
            '[TARGET]类型', # path
        ],
        [
            # 业务环节-负责角色
            yewuhuanjie,
            ["{}部门负责的有哪些业务？", "{}负责什么业务？"],
            ['负责角色'],
            [('负责角色', 'attribute')], # ner
            '[TARGET]负责角色', # path
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
            [(':END_ID', 'entity')], # ner
            '[TARGET]包含', # path
        ],
        [
            # 业务流程包含业务环节
            pd.merge(baohan, yewuliucheng, left_on=':START_ID', right_on=':ID', how='inner'),
            ["包含{}的业务环节是什么？", "包含{}的业务环节有哪些？", "什么业务有{}？", "哪些业务有{}？"],
            [':END_ID'],
            [(':END_ID', 'entity')], # ner
            '[TARGET]包含', # path
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
            [(':START_ID', 'entity'), ('风险等级', 'attribute')], # ner
            '包含[TARGET]风险等级', # path
        ]
    ]