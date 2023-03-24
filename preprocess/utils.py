import pandas as pd
import os
import json
import requests
import random
import string

OPENAI_URL = "https://api.openai-proxy.com/v1/chat/completions"
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
RANDOM_FARM = string.ascii_lowercase + string.digits

def do_request(headers, payload, n_extend):
    all_questions = []
    retry = 0
    while(len(all_questions) != n_extend) and retry < 10:
        all_questions.clear()
        try:
            raw = requests.request("POST", OPENAI_URL, headers=headers, data=payload, timeout=60)
        except requests.ReadTimeout:
            print("Timeout")
            raw = None

        if raw is not None:
            try:
                response = eval(raw.text)
            except:
                response = {'code':-1, 'message': raw.text}
            print(eval(payload)["content"])
            print(response)
            if response['code'] != 200:
                print(f"Request failed for with {response['code']} {response['message']}")
                retry += 1
                continue
            
            if '、' in response['data'] and '\n' not in response['data']:
                all_questions.append(response['data'].split('、'))
            else:
                for question in response['data'].split('\n'):
                    if question == '':
                        continue
                    all_questions.append(question.split(' ')[-1])
        
        if raw is None or all_questions is None or len(all_questions) != n_extend:
            retry += 1
            print(f"Retrying with invalid output {all_questions}")
    if retry == 5:
        print(f"!!! Not able to generate valid output for {payload}")
    return all_questions

def get_synonyms(sentence, n_extend, input_type='word'):
    all_questions = []
    sessionId = ''.join(random.choice(RANDOM_FARM) for i in range(20))
    question = [
        {
            'sentence': f"原句：{sentence}\n输出{min(n_extend, 10)}个中文同义句，省略原句中的部分信息，尽量缩短句子：",
            'word': f"输出“{sentence}”的{min(n_extend, 10)}个中文同义词，每行输出一个："
        },
        {
            'sentence': f"再输出\"{sentence}\"的{min(n_extend, 10)}个中文同义句，省略原句中的部分信息，尽量缩短句子：",
            'word': f"再输出“{sentence}”的{min(n_extend, 10)}个中文同义词，每行输出一个："
        }
    ]
    payload = json.dumps({
        "apiKey": os.getenv("OPENAI_API_KEY"),
        "sessionId": sessionId,
        "content": question[0][input_type]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    all_questions.extend(do_request(headers, payload, min(n_extend, 10)))
    n_extend -= 10
    while n_extend > 0:
        payload = json.dumps({
            "apiKey": os.getenv("OPENAI_API_KEY"),
            "sessionId": sessionId,
            "content": question[1][input_type]
        })
        all_questions.extend(do_request(headers, payload, min(n_extend, 10)))
        n_extend -= 10
    return all_questions

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

        # TaA
        [
            # 风险点-风险等级
            'TaA',
            fengxiandian, # charts
            ["风险等级为{}的风险点有哪些？", "风险等级为{}的风险点有什么？", "{}级别的风险点有哪些？"], # questions
            ['风险等级'], # question slot fills
            [('风险等级', 'attribute')], # ner
            '[TARGET]风险等级', # path
        ],
        [
            # 业务环节-负责角色
            'TaA',
            yewuhuanjie,
            ["{}部门负责的有哪些业务？", "{}负责什么业务？"],
            ['负责角色'],
            [('负责角色', 'attribute')], # ner
            '[TARGET]负责角色', # path
        ],
        # [
        #     # x-含义，预警-x色预警
        # ],

        # TeE
        [
            # 业务环节包含风险
            'TeE',
            pd.merge(baohan, yewuhuanjie, left_on=':START_ID', right_on=':ID', how='inner'),
            ["包含{}的业务环节是什么？", "包含{}的业务环节有哪些？", "什么业务有{}？", "哪些业务有{}？"],
            [':END_ID'],
            [(':END_ID', 'entity')], # ner
            '[TARGET]包含', # path
        ],
        [
            # 业务流程包含业务环节
            'TeE',
            pd.merge(baohan, yewuliucheng, left_on=':START_ID', right_on=':ID', how='inner'),
            ["包含{}的业务环节是什么？", "包含{}的业务环节有哪些？", "什么业务有{}？", "哪些业务有{}？"],
            [':END_ID'],
            [(':END_ID', 'entity')], # ner
            '[TARGET]包含', # path
        ],

        # EeNaT
        [
            # 业务流程包含业务环节-负责角色
            'EeNaT',
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
            'EeNaT',
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                yewuhuanjie,
                left_on=':END_ID', right_on=':ID',
            ),
            ["{}包含的业务环节由谁负责？", "{}包含的业务环节是哪个部门负责的？", 
            "{}各环节由谁负责？", "{}由哪几个部门负责？",
            "{}是哪个部门负责的？", "谁负责{}？", "哪些部门负责{}？"],
            [':START_ID'],
            [(':START_ID', 'entity')], # ner
            '包含[NEDGE]负责角色[TARGET]', # path
        ],
        [
            # 业务环节包含风险点-风险等级
            'EeNaT',
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
            'EeNeT',
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

        # EeTaA
        [
            # 业务流程-[业务环节]-负责角色
            'EeTaA',
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                yewuhuanjie, left_on=':END_ID', right_on=':ID', how='inner'
            ),
            ["{}业务中由{}负责的有哪些？", "{}里哪些业务由{}负责？", "{}中哪几个环节由{}负责"],
            [':START_ID', '负责角色'],
            [(':START_ID', 'entity'), ('负责角色', 'attribute')], # ner
            '包含[TARGET]负责角色', # path
        ],
        [
            # 业务流程-[业务环节]-负责角色
            'EeTaA',
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                yewuhuanjie, left_on=':END_ID', right_on=':ID', how='inner'
            ),
            ["{}负责{}里的哪些业务？", "{}参与{}里哪些业务？", "{}负责{}中哪些部分？", "{}负责哪个{}环节？"],
            ['负责角色', ':START_ID'],
            [(':START_ID', 'entity'), ('负责角色', 'attribute')], # ner
            '包含[TARGET]负责角色', # path
        ],
        [
            # 业务环节-[风险]-风险等级
            'EeTaA',
            pd.merge(
                pd.merge(yewuhuanjie, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                fengxiandian, left_on=':END_ID', right_on=':ID', how='inner'
            ),
            ["{}业务包含的{}等级风险有哪些？", "{}包含{}等级的风险有哪些？", "{}有哪些{}级风险？"],
            [':START_ID', '风险等级'],
            [(':START_ID', 'entity'), ('风险等级', 'attribute')], # ner
            '包含[TARGET]风险等级', # path
        ],

        # EeTeE
        [
            # 业务流程-[业务环节]-风险点
            'EeTeE',
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                fengxiandian, left_on=':END_ID', right_on=':ID', how='inner'
            ),
            ["{}中哪些业务包含{}风险？", "哪些{}工作有{}风险点？", 
            "{}里什么业务有{}风险？", "{}流程里哪些业务有{}风险？"],
            [':START_ID', ':ID_y'],
            [(':START_ID', 'entity'), (':ID_y', 'entity')], # ner
            '包含[TARGET]包含', # path
        ],

        # TeNaA
        [
            # [业务流程]-业务环节-负责角色
            'TeNaA',
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                yewuhuanjie, left_on=':END_ID', right_on=':ID', how='inner'
            ),
            ["哪些流程由{}负责？", "{}负责哪些业务流程？", 
            "{}参与哪些业务流程？", "哪些业务流程里有{}参与？"],
            ['负责角色'],
            [('负责角色', 'attribute')], # ner
            '[TARGET]包含[NEGDE]负责角色', # path
        ],
        [
            # [业务环节]-风险点-风险等级
            'TeNaA',
            pd.merge(
                pd.merge(yewuhuanjie, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                fengxiandian, left_on=':END_ID', right_on=':ID', how='inner'
            ),
            ["哪些业务有{}级风险？", "哪些业务包含{}等级风险点？", 
            "什么业务有{}等级风险", "哪些业务会有{}级风险"],
            ['风险等级'],
            [('风险等级', 'attribute')], # ner
            '[TARGET]包含[NEGDE]风险等级', # path
        ],
        # [
        #     # [业务流程]-风险点-含义
        # ]
        # TeNeE
        [
            # [业务流程]-业务环节-风险点
            'TeNeE',
            pd.merge(
                pd.merge(yewuliucheng, baohan, left_on=':ID', right_on=':START_ID', how='inner'),
                baohan, left_on=':END_ID', right_on=':START_ID', how='inner'
            ),
            ["哪些业务流程包含{}风险？", "哪些业务流程可能有{}风险？",
            "什么业务流程会有{}风险？", "哪个业务流程有{}风险"],
            [':END_ID_y'],
            [(':END_ID_y', 'entity')], # ner
            '[TARGET]包含[NEDGE]包含', # path
        ]
    ]