from .EL_model import EL
import argparse

from config import el_model_config, neo4j_config
from utils import merge_arg_and_config, convert_config_paths, Neo4jGraph

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='input/data/el/train.txt', help='EL模型训练集路径')
    parser.add_argument('--dev_file', default='input/data/el/dev.txt', help='EL模型验证集路径')
    parser.add_argument('--output_dir', default='models/el/el_output/', help='训练好的实体链接模型保存路径')
    config = el_model_config.get("train")
    args = parser.parse_args()
    convert_config_paths(config)
    merge_arg_and_config(args, config)
    return args

def main():
    args = parse_args()
    graph = Neo4jGraph(neo4j_config["neo4j_addr"], neo4j_config["username"], neo4j_config["password"])
    model = EL(args, graph)
    model.train(args)

if __name__ == '__main__':
    main()
