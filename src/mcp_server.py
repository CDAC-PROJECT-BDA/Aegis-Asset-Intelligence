import sqlite3
import pandas as pd
from mcp.server.fastmcp import FastMCP
import chromadb
from sentence_transformers import SentenceTransformer


mcp = FastMCP("AegisAssetIntelligence")

DB_PATH = 'iot_telemetry.db'
CHROMA_PATH = 'chroma_db'

@mcp.tool()
def get_sensor_history(limit: int = 10) -> str:
    """
    Retrieves the most recent sensor history from the live IoT database.
    Useful for checking the recent trends leading up to a failure.
    
    Args:
        limit: Number of recent records to retrieve (max 50)
    """
    try:
        limit = min(limit, 50)
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM telemetry ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        
        if df.empty:
            return "No sensor data found in history."
            
        return df.to_string()
    except Exception as e:
        return f"Error retrieving sensor history: {str(e)}"

@mcp.tool()
def get_manual_section(query: str) -> str:
    """
    Searches the RAG vector database (equipment manuals) for troubleshooting steps.
    
    Args:
        query: The failure description or symptom to look up.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name="pump_manuals")
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        if not results['documents'][0]:
            return "No relevant manual sections found."
            
        combined_docs = "\n\n---\n\n".join(results['documents'][0])
        return f"Extracted Manual Sections for '{query}':\n\n{combined_docs}"
    except Exception as e:
        return f"Error searching manuals: {str(e)}"

@mcp.tool()
def log_maintenance_ticket(component: str, failure_reason: str, priority: str) -> str:
    """
    Logs a maintenance ticket to the CMMS (simulated).
    
    Args:
        component: The part that failed (e.g., 'Thrust Bearing')
        failure_reason: Description of the failure
        priority: 'HIGH', 'MEDIUM', or 'LOW'
    """
    ticket_id = f"TKT-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    return f"SUCCESS: Logged maintenance ticket {ticket_id} for {component}. Priority: {priority}. Reason: {failure_reason}"

if __name__ == "__main__":
    mcp.run()
