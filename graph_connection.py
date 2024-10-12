from langchain_community.graphs import Neo4jGraph

def connect_to_graph(url, username, password):
    graph = Neo4jGraph(url=url, username=username, password=password)
    return graph
