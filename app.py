import os
import json
import streamlit as st
from dotenv import load_dotenv
from graph_connection import connect_to_graph
from data_import import import_data
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import GraphCypherQAChain
from langchain_community.vectorstores import Neo4jVector

# Load environment variables
load_dotenv()

# Neo4j credentials
url = "neo4j+s://c786584f.databases.neo4j.io"
username = "neo4j"
password = "your neo4j pswd here"
api_key = os.getenv('GEMINI_API_KEY')
vector_store_file = "vector_store.json"  # File to store vector store parameters

def save_vector_store(index_params):
    """Save the index parameters needed to recreate the vector store."""
    with open(vector_store_file, 'w') as f:
        json.dump(index_params, f)

def load_vector_store():
    """Load the vector store parameters from a file and recreate it."""
    with open(vector_store_file, 'r') as f:
        params = json.load(f)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key
    )
    
    return Neo4jVector.from_existing_graph(
        url=params['url'],
        username=params['username'],
        password=params['password'],
        index_name=params['index_name'],
        node_label=params['node_label'],
        text_node_properties=params['text_node_properties'],
        embedding=embedding_model,
        embedding_node_property='embedding'
    )

def main():
    """Main function to run the Streamlit app."""
    st.title("RAG-Based Knowledge Graph with Q&A")

    # Step 1: Connect to Neo4j Graph
    if 'graph' not in st.session_state:
        st.write("Connecting to Neo4j...")
        st.session_state['graph'] = connect_to_graph(url, username, password)
        st.success("Connected to Neo4j!")

    graph = st.session_state['graph']

    # Initialize session state variables
    if 'vector_index' not in st.session_state:
        st.session_state['vector_index'] = None
        st.session_state['is_ready'] = False
        st.session_state['use_existing'] = None

    # Step 2: Check for Existing Vector Store
    if os.path.isfile(vector_store_file):
        if 'use_existing' not in st.session_state:  # Check if not already set
            st.session_state['use_existing'] = st.radio(
                "Previous data found. Do you want to use it or import new data?",
                ("Use Existing Data", "Import New Data"),
                index=None  # Ensure no option is selected initially
            )

        if st.session_state['use_existing'] == "Use Existing Data":
            st.session_state['vector_index'] = load_vector_store()
            st.success("Vector store loaded successfully! You can now ask questions.")
            st.session_state['is_ready'] = True  # Set to true when ready for Q&A

        elif st.session_state['use_existing'] == "Import New Data":
            st.error("You chose to import new data. Please provide the URL to import data.")
            import_url = st.text_input("Enter JSON Data Import URL (only JSON format allowed)")

            if st.button("Import Data"):
                if import_url:
                    import_data(graph, import_url)
                    st.success("Data Imported Successfully!")

                    embedding_model = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004", 
                        google_api_key=api_key
                    )
                    
                    st.session_state['vector_index'] = Neo4jVector.from_existing_graph(
                        url=url, 
                        username=username, 
                        password=password,
                        index_name='tasks',
                        node_label="Task",
                        text_node_properties=['name', 'description', 'status'],
                        embedding=embedding_model,
                        embedding_node_property='embedding'
                    )

                    save_vector_store({
                        'url': url,
                        'username': username,
                        'password': password,
                        'index_name': 'tasks',
                        'node_label': 'Task',
                        'text_node_properties': ['name', 'description', 'status']
                    })
                    st.success("Embeddings created and data stored locally!")
                    st.session_state['is_ready'] = True  # Set to true when ready for Q&A

    else:
        st.error("No existing data found. Please provide the URL to import data.")
        import_url = st.text_input("Enter JSON Data Import URL (only JSON format allowed)")

        if st.button("Import Data"):
            if import_url:
                import_data(graph, import_url)
                st.success("Data Imported Successfully!")

                embedding_model = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004", 
                    google_api_key=api_key
                )
                
                st.session_state['vector_index'] = Neo4jVector.from_existing_graph(
                    url=url, 
                    username=username, 
                    password=password,
                    index_name='tasks',
                    node_label="Task",
                    text_node_properties=['name', 'description', 'status'],
                    embedding=embedding_model,
                    embedding_node_property='embedding'
                )

                save_vector_store({
                    'url': url,
                    'username': username,
                    'password': password,
                    'index_name': 'tasks',
                    'node_label': 'Task',
                    'text_node_properties': ['name', 'description', 'status']
                })
                st.success("Embeddings created and data stored locally!")
                st.session_state['is_ready'] = True  # Set to true when ready for Q&A

    # Step 3: Ask Questions using the Dataset
    if st.session_state['is_ready']:
        run_qna()

def run_qna():
    """Initialize the Q&A system if the vector index is ready and handle user questions."""
    # Initialize LLM and Q&A only once
    if 'llm' not in st.session_state:
        st.session_state['llm'] = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro", 
            google_api_key=api_key
        )
        st.session_state['vector_qa'] = GraphCypherQAChain.from_llm(
            cypher_llm=st.session_state['llm'],
            qa_llm=st.session_state['llm'],
            graph=st.session_state['graph'], 
            verbose=True
        )
        st.success("Q&A System Ready!")

    # Step 4: Ask Questions from Your Dataset
    st.header("Ask Questions from Your Dataset")
    question = st.text_input("Enter your question:")
    clicked = st.button("Ask")

    if clicked:  # Only execute when button is clicked
        if question:
            response = st.session_state['vector_qa'].invoke({"query": question})
            st.write("Answer:", response['result'])
            if 'questions_asked' not in st.session_state:
                st.session_state['questions_asked'] = []  # Initialize if not present
            st.session_state['questions_asked'].append(question)  # Store the question
        else:
            st.warning("Please enter a question before clicking 'Ask'.")

if __name__ == "__main__":
    main()
