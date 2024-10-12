from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def setup_embeddings(url, username, password, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        url=url,
        username=username,
        password=password,
        index_name='tasks',
        node_label="Task",
        text_node_properties=['name', 'description', 'status'],
        embedding_node_property='embedding',
    )
    return vector_index
