from py2neo import Graph, Node, NodeMatcher

class Neo4jGraph():
    
    def __init__(self, remote_addr, username, password):
        self.graph = Graph(remote_addr, auth=(username, password))
        self._build_entity_list()

    def execute_query(self, query):
        dict_list = self.graph.run(query).data()
        return sum([list(d.values()) for d in dict_list], [])

    def _build_entity_list(self):
        get_all_e_query = 'MATCH (n) WHERE EXISTS(n.`名称`) RETURN DISTINCT n.`名称`'
        self.entity_list = sorted(self.execute_query(get_all_e_query), key=lambda x: len(x), reverse=True)