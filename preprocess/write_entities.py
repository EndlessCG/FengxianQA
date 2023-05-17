from config import neo4j_config
from utils import Neo4jGraph

# 属性值不写到synonyms.txt中的属性列表
ATTRIBUTE_BLACKLIST = ['含义', '橙色预警', '蓝色预警', '红色预警', '黄色预警']
# 输出路径
TARGET_PATH = 'preprocess/synonyms_template.txt'

def main():
    graph = Neo4jGraph(neo4j_config["neo4j_addr"], neo4j_config["username"], neo4j_config["password"])
    all_words = graph.get_all_words(attr_blacklist=ATTRIBUTE_BLACKLIST)
    with open(TARGET_PATH, 'w+') as f:
        f.writelines('\n'.join(all_words))

if __name__ == '__main__':
    main()