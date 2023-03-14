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
            'EaT',
            all_nodes,
            ["{}的含义是什么？", "{}的定义是什么？", "{}是什么意思？", "{}代表什么？", "{}指什么？", "怎么解释{}？", "怎么理解{}？"],
            [':ID'],
            [(':ID', 'entity')],
            '含义[TARGET]',
        ],
        [
            # 风险等级
            'EaT',
            all_nodes,
            ["{}的风险等级是多少？", "{}是什么等级的风险？"],
            [':ID'],
            [(':ID', 'entity')],
            '风险等级[TARGET]',
        ],
        [
            # 类型
            'EaT',
            all_nodes,
            ["{}属于什么类型？", "{}属于什么类别？", "{}的类型是什么？", "{}的类别是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '类型[TARGET]',
        ],
        [
            # 业务环节-负责角色
            'EaT',
            yewuhuanjie,
            ["{}是哪个部门负责的？", "谁负责{}？"],
            [':ID'],
            [(':ID', 'entity')],
            '负责角色[TARGET]',
        ],
        [
            # 红色预警
            'EaT',
            yujing,
            ["{}在什么时候会发生红色预警？", "{}的红色预警是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '红色预警[TARGET]',
        ],
        [
            # 橙色预警
            'EaT',
            yujing,
            ["{}在什么时候会发生橙色预警？", "{}的橙色预警是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '橙色预警[TARGET]',
        ],
        [
            # 黄色预警
            'EaT',
            yujing,
            ["{}在什么时候会发生黄色预警？", "{}的黄色预警是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '黄色预警[TARGET]',
        ],
        [
            # 蓝色预警
            'EaT',
            yujing,
            ["{}在什么时候会发生蓝色预警？", "{}的蓝色预警是什么？"],
            [':ID'],
            [(':ID', 'entity')],
            '蓝色预警[TARGET]',
        ],

        # EeT
        [
            # 包含
            'EeT',
            baohan,
            ["{}包含哪些内容？", "{}包括哪些内容？", "{}涉及哪些内容？"],
            [':START_ID'],
            [(':START_ID', 'entity')],
            '包含[TARGET]'
        ],
        [
            # 业务流程包含业务环节
            'EeT',
            baohan,
            ["{}包含哪些环节？", "{}包括哪些环节？", "{}有哪些环节？", "{}有哪些步骤？"],
            [':START_ID'],
            [(':START_ID', 'entity')],
            '包含[TARGET]'
        ],
        [
            # 业务环节包含风险点
            'EeT',
            baohan,
            ["{}包含哪些风险点？", "{}包括什么风险？", "{}包含什么风险？", "{}有什么风险？", "{}有什么危险？", "{}可能有什么危险？"],
            [':START_ID'],
            [(':START_ID', 'entity')],
            '包含[TARGET]'
        ],
    ]