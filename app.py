import os
import streamlit as st
import google.generativeai as genai
import psycopg2
from contextlib import contextmanager
from supabase.client import create_client, Client 

# --- LangChain/RAG Imports ---
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings 
# ====================================================================
# 🔑 1. CONFIGURATION & DATABASE FUNCTIONS
# ====================================================================

# 🔑 Configure Gemini API
try:
    # Uses st.secrets.GOOGLE_API_KEY from .streamlit/secrets.toml
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except Exception:
    st.error("🚨 GEMINI_API_KEY not found in secrets.toml.")

# --- Supabase PostgreSQL Connection (For Sessions & Chat History) ---
@contextmanager
def get_db_connection():
    """Establishes connection to the Supabase PostgreSQL database."""
    conn = None
    try:
        # Uses st.secrets.postgres details
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            port=6543,
            sslmode=st.secrets["postgres"]["sslmode"] 
            
        )
        yield conn
    except Exception as e:
        st.error(f"🚨 Database connection failed! Check secrets.toml: {e}")
        raise
    finally:
        if conn:
            conn.close()

# ====================================================================
# 🧠 2. RAG/KNOWLEDGE BASE SETUP (Online pgvector)
# ====================================================================

@st.cache_resource
def setup_knowledge_base_online():
    """Connects to Supabase pgvector and returns the RAG retriever."""
    st.info("🌐 Connecting to online knowledge base...")
    try:
        # Initialize Supabase Client using secrets
        supabase_client: Client = create_client(
            st.secrets["supabase"]["url"], 
            st.secrets["supabase"]["key"]
        )
        
        # Initialize the same embeddings model used for ingestion
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Connect to the remote vector store
        vectorstore = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase_client,
            table_name="iitb_knowledge", 
            query_name="match_documents" 
        )

        st.success("✅ Knowledge base loaded from Supabase pgvector.")
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"🚨 Error setting up RAG knowledge base: {e}")
        return None

# ====================================================================
# ⚙️ 3. SESSION MANAGEMENT & HISTORY LOGIC
# ====================================================================

def create_new_anonymous_session():
    """Creates a new unique session_id in the DB and updates session_state."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            # Inserts a new row and returns the new session_id
            cur.execute("INSERT INTO chat_sessions DEFAULT VALUES RETURNING session_id")
            new_id = cur.fetchone()[0]
            conn.commit()
            
            st.session_state.current_session_id = new_id
            st.session_state.messages = [] # Clear displayed history
            st.session_state.query_count = 0 # Reset query count
            st.toast(f"Starting new chat thread (ID: {new_id})", icon='💬')

    except Exception as e:
        st.error(f"Could not initialize session: {e}. Chat history will not be saved.")
        st.session_state.current_session_id = -1 # Sentinel value
        st.session_state.messages = []

def load_session_history(session_id):
    """Loads chat history and query count for a given session ID."""
    if session_id == -1: return

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            # 1. Load messages
            cur.execute(
                "SELECT user_query, bot_response FROM chat_messages WHERE session_id = %s ORDER BY timestamp ASC",
                (session_id,)
            )
            rows = cur.fetchall()
            
            # 2. Count queries
            cur.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = %s",
                (session_id,)
            )
            count = cur.fetchone()[0]

            # Update session state
            st.session_state.messages = []
            for user_msg, bot_msg in rows:
                st.session_state.messages.append({"role": "user", "content": user_msg})
                st.session_state.messages.append({"role": "assistant", "content": bot_msg})
            
            st.session_state.current_session_id = session_id
            st.session_state.query_count = count
            st.toast(f"Loaded session ID: {session_id} ({count} queries asked)")

    except Exception as e:
        st.error(f"Could not load chat history for session {session_id}.")
        st.session_state.messages = []


def save_message_to_db(session_id, query, answer):
    """Saves the Q&A to the chat_messages table."""
    if session_id == -1: return

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO chat_messages (session_id, user_query, bot_response) VALUES (%s, %s, %s)",
                (session_id, query, answer)
            )
            conn.commit()
            st.session_state.query_count += 1
    except Exception as e:
        st.error(f"Could not save chat message to database: {e}")

# ====================================================================
# 🖼️ 4. STREAMLIT UI LAYOUT
# ====================================================================

st.set_page_config(page_title="CampusGPT", page_icon="🎓", layout="centered")
st.title("🎓 CampusGPT (IITB Anonymous Chat)")
st.write("Ask questions about IIT Bombay (RAG powered by Gemini)")

# --- Initialize Session State ---
if "current_session_id" not in st.session_state:
    create_new_anonymous_session()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

retriever = setup_knowledge_base_online()
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

# --- Sidebar: All Sessions Ever Asked (Requirement #6) ---
with st.sidebar:
    st.header("Chat Threads")
    if st.button("➕ Start New Chat Thread", use_container_width=True):
        create_new_anonymous_session()
    
    st.markdown("---")
    st.subheader("History (All Users)")

    # Fetch all recent sessions/first messages
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            # Fetch the first message and the last activity timestamp for all sessions
            cur.execute("""
                WITH latest_activity AS (
                    SELECT 
                        session_id, 
                        MAX(timestamp) as last_activity
                    FROM chat_messages
                    GROUP BY session_id
                )
                SELECT 
                    cs.session_id, 
                    COALESCE(
                        (SELECT user_query FROM chat_messages WHERE session_id = cs.session_id ORDER BY timestamp ASC LIMIT 1), 
                        'New Session'
                    ) AS first_query
                FROM chat_sessions cs
                LEFT JOIN latest_activity la ON cs.session_id = la.session_id
                ORDER BY la.last_activity DESC NULLS LAST 
                LIMIT 15;
            """)
            sessions = cur.fetchall()
    except Exception:
        sessions = []
        st.warning("Cannot load session list from DB.")

    # Display sessions as selectable buttons/radio
    for session_id, first_query in sessions:
        # Truncate the query for display
        display_query = f"Thread {session_id}: {first_query[:35]}..." if len(first_query) > 35 else f"Thread {session_id}: {first_query}"
        
        # Use a button to select the session
        if st.sidebar.button(
            display_query, 
            key=f"session_btn_{session_id}",
            type="primary" if session_id == st.session_state.current_session_id else "secondary",
            use_container_width=True
        ):
            # Only reload if clicking a different session
            if session_id != st.session_state.current_session_id:
                load_session_history(session_id)

    st.markdown("---")
    st.caption(f"Current Thread ID: **{st.session_state.current_session_id}**")


# --- Main Chat Display (Requirement #7) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---
query_limit_reached = st.session_state.query_count >= 10
placeholder_text = (
    "Ask your question here..." 
    if not query_limit_reached else 
    f"🛑 Session limit ({st.session_state.query_count}/10) reached. Start a new chat thread from the sidebar."
)

if query := st.chat_input(placeholder_text, disabled=query_limit_reached):
    
    # 1. Display user query
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        # 2. RAG/Gemini Logic
        context = ""
        if retriever:
            # Query pgvector (online)
            results = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in results])

        prompt = (
            f"You are CampusGPT, an assistant for IIT Bombay. Use the context below to answer the question. "
            f"If the context is irrelevant or not provided, answer as a helpful assistant with general knowledge, "
            f"but always prioritize the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        
        try:
            response = gemini_model.generate_content(contents=prompt)
            answer = response.text
        except Exception as e:
            answer = f"Sorry, I encountered an error with the AI model: {e}"

    # 3. Display bot answer
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # 4. Save to Supabase and update counter
    save_message_to_db(st.session_state.current_session_id, query, answer)
    # Rerun the app to update the sidebar and display the query count check
    st.rerun()