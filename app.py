import os
from datetime import datetime
import pandas as pd
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from crewai.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain Imports for Multi-PDF FAISS RAG
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Support Desk",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------
# ENVIRONMENT CONFIGURATION (Strictly from .env file)
# ---------------------------------------------------------
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
serper_api_key = os.environ.get("SERPER_API_KEY", "")

# ---------------------------------------------------------
# SIDEBAR: KNOWLEDGE BASE MANAGEMENT
# ---------------------------------------------------------
st.sidebar.header("📁 Knowledge Base Management")
PDF_DIR = "./uploaded_pdfs"
FAISS_INDEX_DIR = "faiss_index"
os.makedirs(PDF_DIR, exist_ok=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload multiple PDF documents", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

@st.cache_resource
def initialize_multi_pdf_faiss():
    """Loads all PDFs from directory, creates embeddings, and builds FAISS index."""
    if not openai_api_key:
        return None
        
    embeddings = OpenAIEmbeddings()
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        if os.path.exists(FAISS_INDEX_DIR):
            return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return None
    
    loader = PyPDFDirectoryLoader(PDF_DIR)
    pages = loader.load()
    
    if not pages:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)
    
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    
    return vector_store

# ---------------------------------------------------------
# CUSTOM TOOLS DEFINITION
# ---------------------------------------------------------
@tool("RAG Knowledge Base Retrieval")
def rag_retrieval_tool(query: str) -> str:
    """
    Searches across all internal company PDF documents in the FAISS vector store for an answer.
    """
    if not os.path.exists(FAISS_INDEX_DIR):
        return "NOT_FOUND: FAISS vector index not initialized."
    
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
            FAISS_INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        results = vector_store.similarity_search(query, k=4)
        
        if not results:
            return "NOT_FOUND"
            
        context = "\n\n".join([f"\n{doc.page_content}" for doc in results])
        return context
    except Exception as e:
        return "NOT_FOUND"


@tool("Excel Logger")
def log_to_excel(user: str, question: str, answer: str) -> str:
    """
    Logs user interaction details (user, Question, answer, and time) into an Excel file.
    """
    file_path = "Enquiry.xlsx"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_data = {
        "user": [user],
        "Question": [question],
        "answer": [answer],
        "time": [current_time]
    }
    
    df_new = pd.DataFrame(new_data)
    
    if os.path.exists(file_path):
        df_existing = pd.read_excel(file_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(file_path, index=False)
    else:
        df_new.to_excel(file_path, index=False)
        
    return f"Successfully logged interaction for {user} at {current_time}."


# ---------------------------------------------------------
# CREWAI EXECUTION FUNCTION
# ---------------------------------------------------------
def run_support_crew(user_name: str, customer_question: str):
    rag_agent = Agent(
        role="Internal Knowledge Specialist",
        goal="Retrieve accurate answers to customer questions from the multi-PDF FAISS RAG system.",
        backstory="You are an expert at searching across multiple internal company PDF documents via FAISS. If the RAG tool returns 'NOT_FOUND', acknowledge that the internal database has no answer.",
        tools=[rag_retrieval_tool],
        verbose=True
    )

    search_agent = Agent(
        role="Web Research Specialist",
        goal="Search the web for up-to-date and accurate answers when the internal RAG system fails.",
        backstory="You are an expert researcher. When internal documents do not contain the answer, you use web search to find reliable information.",
        tools=[SerperDevTool()],
        verbose=True
    )

    logging_agent = Agent(
        role="Support Data Logger",
        goal="Synthesize the final support answer and log the transaction details into the Excel file.",
        backstory="You ensure every customer interaction is accurately recorded with the username, question, final answer, and timestamp.",
        tools=[log_to_excel],
        verbose=True
    )

    task_rag = Task(
        description=f"Search the internal RAG knowledge base for the following question: '{customer_question}'. If found, output it. If it returns 'NOT_FOUND', explicitly state 'NOT_FOUND'.",
        expected_output="The retrieved answer from the RAG system or 'NOT_FOUND'.",
        agent=rag_agent
    )

    task_search = Task(
        description=f"Review the previous task output. If it says 'NOT_FOUND', perform a web search to find an accurate answer for: '{customer_question}'. If RAG provided a valid answer, pass it through.",
        expected_output="The final satisfactory answer sourced either from RAG or web search.",
        agent=search_agent
    )

    task_log = Task(
        description=f"Take the final answer from the previous task and use the Excel Logger tool to record the entry. User: '{user_name}', Question: '{customer_question}', Answer: [Final Answer].",
        expected_output="Confirmation that the entry has been successfully logged to Excel.",
        agent=logging_agent
    )

    support_crew = Crew(
        agents=[rag_agent, search_agent, logging_agent],
        tasks=[task_rag, task_search, task_log],
        process=Process.sequential,
        verbose=True
    )

    return support_crew.kickoff()


# ---------------------------------------------------------
# STREAMLIT UI LAYOUT
# ---------------------------------------------------------
st.title("Customer Support Portal")
st.markdown("Powered by **CrewAI**, **FAISS Multi-PDF RAG**, **Serper Search**, and **Excel Logging**.")

tab1, tab2 = st.tabs(["💬 Interactive Chat Support", "📊 View Interaction Logs"])

with tab1:
    st.subheader("Customer Support Chat")
    
    if not openai_api_key or not serper_api_key:
        st.error("⚠️ API keys missing! Please ensure both `OPENAI_API_KEY` and `SERPER_API_KEY` are defined in your local `.env` file.")
    else:
        with st.spinner("Checking and updating FAISS vector index from PDFs..."):
            vector_db = initialize_multi_pdf_faiss()
            
        if vector_db:
            st.success("Multi-PDF Knowledge Base loaded and indexed successfully!")
        else:
            st.info("ℹ️ Ready. Upload PDF files in the sidebar if you have internal documents.")

        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Customer name input for logging
        user_name = st.text_input("Customer Name for this session", value="Valued Customer", placeholder="e.g., John Doe")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input via chat input box
        if customer_question := st.chat_input("Ask your question here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": customer_question})
            with st.chat_message("user"):
                st.markdown(customer_question)

            # Generate response using the multi-agent crew
            with st.chat_message("assistant"):
                with st.spinner("Processing ..."):
                    try:
                        final_response = run_support_crew(user_name, customer_question)
                        response_text = str(final_response)
                        st.markdown(response_text)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    except Exception as e:
                        error_msg = f"An error occurred during crew execution: {e}"
                        st.error(error_msg)

with tab2:
    st.subheader("Support Logs Database (`Enquiry.xlsx`)")
    
    if os.path.exists("Enquiry.xlsx"):
        df_logs = pd.read_excel("Enquiry.xlsx")
        st.dataframe(df_logs, use_container_width=True)
        
        with open("Enquiry.xlsx", "rb") as f:
            st.download_button(
                label="📥 Download Excel Logs",
                data=f,
                file_name="Enquiry.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("No interaction logs found yet.")