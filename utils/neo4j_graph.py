from py2neo import Graph

class Neo4jGraph():
    
    def __init__(self, remote_addr, username, password):
        self.graph = Graph(remote_addr, auth=(username, password))
        self._build_entity_lists()

    def execute_query(self, query, drop_keys=True):
        dict_list = self.graph.run(query).data()
        if drop_keys:
            if len(dict_list) != 0 and len(dict_list[0]) != 1:
                return [list(d.values()) for d in dict_list]
            else:
                return sum([list(d.values()) for d in dict_list], [])
        else:
            return dict_list

    def _build_entity_lists(self):
        get_all_e_query = 'MATCH (n) WHERE EXISTS(n.`名称`) RETURN DISTINCT n.`名称`'
        get_all_a_query = 'match (n) return [k IN KEYS(n) | n[k]]'
        self.entity_list = sorted(self.execute_query(get_all_e_query), key=lambda x: len(x), reverse=True)
        self.attribute_list = self.execute_query(get_all_a_query)
        self.attribute_list = sum(self.attribute_list, [])
        self.attribute_list = [i if isinstance(l, list) else l for l in self.attribute_list for i in l]
        self.attribute_list = list(set(self.attribute_list))
        self.attribute_list = sorted(self.attribute_list, key=lambda x: len(x), reverse=True)
