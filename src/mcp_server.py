import sqlite3
import pandas as pd
import os
import sys


os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


original_stdout = sys.stdout
original_stderr = sys.stderr


sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

try:
    import chromadb
    from sentence_transformers import SentenceTransformer

    CHROMA_PATH = "chroma_db"
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    manual_collection = chroma_client.get_collection(name="pump_manuals")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    
    
    embedding_model.encode(["warmup"])

    import logging
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("fastmcp").setLevel(logging.ERROR)
finally:
    
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr

from fastmcp import FastMCP


mcp = FastMCP("AegisAssetIntelligence")

DB_PATH = 'iot_telemetry.db'
CHROMA_PATH = 'chroma_db'

@mcp.tool()
def get_sensor_history(limit: int = 5) -> str:
    """
    Retrieves the most recent sensor history from the live IoT database.
    Useful for checking the recent trends leading up to a failure.

    Args:
        limit: Number of recent records to retrieve (max 10)
    """
    try:
        limit = min(limit, 10)
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
    Search the pump manual for specific troubleshooting or maintenance information.
    Provide a specific query like 'excessive vibration' or 'how to replace seal'.
    """
    try:
        query_embedding = embedding_model.encode([query]).tolist()
        results = manual_collection.query(
            query_embeddings=query_embedding,
            n_results=1
        )
        if not results['documents'] or not results['documents'][0]:
            return "No relevant manual sections found."
        
        content = "\n\n---\n\n".join(results['documents'][0])
        
        if len(content) > 3000:
            content = content[:3000] + "\n...[TRUNCATED TO SAVE TOKENS]..."
        return content
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
    try:
        mcp.run(show_banner=False)
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        print('MCP server crashed:', e, file=sys.stderr)
