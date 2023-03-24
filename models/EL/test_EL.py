from .EL_model import EL
import argparse

from config import el_model_config, neo4j_config
from utils import merge_arg_and_config, convert_config_paths, Neo4jGraph

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default='input/data/el/test.txt', help='EL模型测试集路径')
    parser.add_argument('--model_path', default='models/EL/el_output/best_el.bin', help='测试的EL模型路径')
    config = el_model_config.get("test")
    args = parser.parse_args()
    convert_config_paths(config)
    merge_arg_and_config(args, config)
    return args

def main():
    args = parse_args()
    graph = Neo4jGraph(neo4j_config["neo4j_addr"], neo4j_config["username"], neo4j_config["password"])
    model = EL(args, graph)
    model.load_model(args.model_path)
    model.test(args)

if __name__ == '__main__':
    main()