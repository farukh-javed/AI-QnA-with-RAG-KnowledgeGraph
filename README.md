# RAG-Based Knowledge Graph with Q&A

This project is a **Retrieval-Augmented Generation (RAG) based Knowledge Graph** built using **Streamlit**, **Neo4j**, **LangChain**, and **Google Generative AI (Gemini-1.5-pro)**. The system allows users to import data into a Neo4j database, generate vector embeddings using Google AI, and ask questions that are answered by combining vector similarity search with the power of Generative AI.

## Key Features

- **Graph Database Integration**: Store, query, and manage your knowledge graph with **Neo4j**.
- **Vector Embeddings**: Generate vector embeddings using **Google Generative AI** for similarity searches.
- **Q&A System**: Ask questions related to the dataset and receive AI-generated responses based on relevant data.
- **Interactive User Interface**: Seamlessly interact with the system through a simple, easy-to-use **Streamlit** app.

## Tech Stack

- **Streamlit**: Frontend for the interactive UI.
- **Neo4j**: Graph database for data storage and querying.
- **LangChain**: Manages embeddings and vector similarity search.
- **Google Generative AI**: Provides embeddings and natural language answers.

## Project Structure

```
rag_knowledge_graph/
├── app.py                     # Main Streamlit app
├── graph_connection.py         # Connects to Neo4j database
├── data_import.py              # Imports data into Neo4j from a URL
├── embedding.py                # Sets up vector embeddings using Google Generative AI
├── requirements.txt            # Python dependencies
└── .env                        # Environment file for API keys
```

## Setup Instructions

### Prerequisites
- Python 3.x
- Neo4j account (or local setup)
- Google Generative AI API key

### Steps to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/farukh-javed/AI-QnA-with-RAG-KnowledgeGraph.git
   cd AI-QnA-with-RAG-KnowledgeGraph
   ```

2. **Create a `.env` file** in the root directory and add your Google API key:
   ```
   GEMINI_API_KEY=your_google_api_key
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

5. **Open the application** in your browser at `http://localhost:8501` and start interacting:
   - Connect to the Neo4j database.
   - Import data from a provided URL.
   - Set up vector embeddings.
   - Ask questions from your dataset.

## Environment Variables

Make sure your `.env` file contains the following:

```
GEMINI_API_KEY=your_google_api_key
```

## License

This project is licensed under the MIT License.

---