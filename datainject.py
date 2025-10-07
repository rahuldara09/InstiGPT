import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# --- CORRECTED IMPORT PATH ---
from langchain_huggingface import HuggingFaceEmbeddings # Install this package: pip install langchain-huggingface
# -----------------------------
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client 
import configparser

# ====================================================================
# 🔑 1. CONFIGURATION 
# ====================================================================

# NOTE: You must install the new package for the correct import: pip install langchain-huggingface

def load_secrets(filepath=".streamlit/secrets.toml"):
    """Manually load Supabase credentials from secrets.toml."""
    parser = configparser.ConfigParser()
    try:
        with open(filepath, 'r') as f:
            content = '[dummy_section]\n' + f.read()
        parser.read_string(content)
        
        supabase_url = parser.get('supabase', 'url').strip('"')
        supabase_key = parser.get('supabase', 'key').strip('"')
        
        return supabase_url, supabase_key
    except Exception as e:
        print(f"Error loading secrets from {filepath}: {e}")
        print("Please ensure .streamlit/secrets.toml exists and has the [supabase] section.")
        exit(1)

# --- Define Constants ---
DATA_DIR = "./data" 
TABLE_NAME = "iitb_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ====================================================================
# 🧠 2. INGESTION FUNCTION
# ====================================================================

def ingest_data_to_supabase():
    """Main function to load, chunk, embed, and upload data."""
    # 1. Load Secrets
    SUPABASE_URL, SUPABASE_KEY = load_secrets()
    
    # 2. Initialize Supabase Client
    print("Initializing Supabase Client...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # 3. Load Documents
    print(f"Loading documents from {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found. Please create it and add your PDFs.")
        exit(1)

    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("No PDFs found in the 'data/' directory. Exiting.")
        exit(1)

    # 4. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks.")

    # 5. Initialize Embeddings Model (No warning now!)
    print(f"Initializing embeddings model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 6. Store Embeddings in Supabase (pgvector)
    print(f"Generating embeddings and uploading to {TABLE_NAME}...")
    
    SupabaseVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        client=supabase,
        table_name=TABLE_NAME,
        query_name="match_documents" 
    )
    
    print("\n\n✅ **Success!** Ingestion complete. The IITB knowledge base is now stored online.")

if __name__ == "__main__":
    ingest_data_to_supabase()