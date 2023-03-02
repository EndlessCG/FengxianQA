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

    all_nodes = pd.concat([fengxiandian, mingci, yewuliucheng, yewuhuanjie, yujing], axis=0, ignore_index=True)

    return [
        # EaT
        [
            # 含义
            all_nodes,
            ["{}的含义是什么？", "{}的定义是什么？", "{}是什么意思？", "{}代表什么？", "{}指什么？", "怎么解释{}？", "怎么理解{}？"],
            [':ID'],
            [(':ID', 'entity')],
            '含义[TARGET]',
        ],
        [
            # 风险等级
            all_nodes,
            ["{}的风险等级是多少？", "{}是什么等级的风险？"],
            [':ID'],
            [(':ID', 'entity')],
            '风险等级[TARGET]',
        ],
        [
            # 类型
            all_nodes,
            ["{}属于什么类型？", "{}属于什么类别？", "{}的类型是什么？", "{}的类别是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '类型[TARGET]',
        ],
        [
            # 负责角色
            pd.concat([fengxiandian, mingci, yewuhuanjie, yujing], ignore_index=True, axis=0),
            ["{}是哪个部门负责的？", "谁负责{}？"],
            [':ID'],
            [(':ID', 'entity')],
            '负责角色[TARGET]',
        ],
        [
            # 红色预警
            all_nodes,
            ["{}在什么时候会发生红色预警？", "{}的红色预警是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '红色预警[TARGET]',
        ],
        [
            # 橙色预警
            all_nodes,
            ["{}在什么时候会发生橙色预警？", "{}的橙色预警是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '橙色预警[TARGET]',
        ],
        [
            # 黄色预警
            all_nodes,
            ["{}在什么时候会发生黄色预警？", "{}的黄色预警是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '黄色预警[TARGET]',
        ],
        [
            # 蓝色预警
            all_nodes,
            ["{}在什么时候会发生蓝色预警？", "{}的蓝色预警是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '蓝色预警[TARGET]',
        ],
        [
            # 预警-类型
            all_nodes,
            ["{}属于什么类型？", "{}属于什么类别？", "{}的类型是什么？", "{}的类别是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '类型[TARGET]',
        ],

        # EeT
        [
            # 包含
            baohan,
            ["{}包含哪些内容？", "{}包括哪些内容？", "{}涉及哪些内容？"],
            [':START_ID'],
            [(':START_ID', 'entity')],
            '包含[TARGET]'
        ],
        [
            # 业务流程包含业务环节
            baohan,
            ["{}包含哪些环节？", "{}包括哪些环节？", "{}有哪些环节？", "{}有哪些步骤？"],
            [':START_ID'],
            [(':START_ID', 'entity')],
            '包含[TARGET]'
        ],
        [
            # 业务环节包含风险点
            baohan,
            ["{}包含哪些风险点？", "{}包括什么风险？", "{}包含什么风险？", "{}有什么风险？", "{}有什么危险？", "{}可能有什么危险？"],
            [':START_ID'],
            [(':START_ID', 'entity')],
            '包含[TARGET]'
        ],

        # EeNaT
        [
            # 业务流程包含业务环节-负责角色
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                yewuhuanjie,
                left_on=':END_ID', right_on=':ID',
            ),
            ["{}包含的业务环节由谁负责？", "{}包含的业务环节是哪个部门负责的？", 
            "{}各环节由谁负责？", 
            "{}是哪个部门负责的？", "谁负责{}？"],
            [':START_ID'],
            [(':START_ID', 'entity')], # ner
            '包含[NEDGE]负责角色[TARGET]', # path
        ],
        [
            # 业务环节包含风险点-含义
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                yewuhuanjie,
                left_on=':END_ID', right_on=':ID',
            ),
            ["{}包含的业务环节由谁负责？", "{}包含的业务环节是哪个部门负责的？", 
            "{}各环节由谁负责？", 
            "{}是哪个部门负责的？", "谁负责{}？"],
            [':START_ID'],
            [(':START_ID', 'entity')], # ner
            '包含[NEDGE]负责角色[TARGET]', # path
        ],
        [
            # 业务环节包含风险点-风险等级
            pd.merge(
                pd.merge(yewuhuanjie, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                fengxiandian,
                left_on=':END_ID', right_on=':ID',
            ),
            ["{}有什么等级的风险？", "{}有哪些等级的风险？", 
            "{}的风险都是什么等级的？"],
            [':START_ID'],
            [(':START_ID', 'entity')], # ner
            '包含[NEDGE]风险等级[TARGET]', # path
        ],

        # EeNeT
        [
            # 业务流程包含业务环节包含风险
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                baohan,
                left_on=':END_ID', right_on=':START_ID',
            ),
            ["{}包含的业务环节有哪些风险？", "{}的各环节包含哪些风险点？", "{}流程包含哪些风险点？", "{}包含哪些风险点？"],
            [':START_ID_x'],
            [(':START_ID_x', 'entity')], # ner
            '包含[NEDGE]包含[TARGET]', # path
        ],

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
            ["{}业务包含的{}等级风险有哪些？", "{}包含{}等级的风险有哪些？", "{}有哪些{}级风险？"],
            [':START_ID', '风险等级'],
            [(':START_ID', 'entity'), ('风险等级', 'attribute')], # ner
            '包含[TARGET]风险等级', # path
        ]
    ]