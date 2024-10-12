import requests

def import_data(graph, import_url):
    import_query = requests.get(import_url).json()['query']
    graph.query(import_query)
