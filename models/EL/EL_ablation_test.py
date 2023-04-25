from .test_EL import parse_args as test_parse_args
from .EL_main import parse_args as train_parse_args
from .EL_model import EL

from config import el_model_config, neo4j_config
from utils import merge_arg_and_config, convert_config_paths, Neo4jGraph

def main():
    train_args = train_parse_args()
    test_args = test_parse_args()
    graph = Neo4jGraph(neo4j_config["neo4j_addr"], neo4j_config["username"], neo4j_config["password"])
    model = EL(train_args, graph)
    model.ablation_test(train_args, test_args)

if __name__ == '__main__':
    main()