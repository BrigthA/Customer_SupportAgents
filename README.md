# 🤖 Multi-Agent Customer Support Portal

An intelligent, multi-agent customer support system built using **CrewAI**, **Streamlit**, **FAISS Vector Store (Multi-PDF RAG)**, **Serper Web Search**, and **Pandas Excel Logging**.

---

## 🚀 Key Features

* **Interactive Streamlit Interface:** Features a real-time chat window for customer inquiries and an administrative tab to view and download live interaction logs.
* **Tiered Multi-Agent Workflow:**
  1. **Internal Knowledge Specialist (Agent 1):** Searches company documentation using a local **FAISS** vector database via LangChain.
  2. **Web Research Specialist (Agent 2):** Automatically falls back to **Serper** web search if the internal RAG system cannot find an answer (`NOT_FOUND`).
  3. **Support Data Logger (Agent 3):** Automatically records every transaction (`user`, `Question`, `answer`, and `time`) into a local Excel file (`support_logs.xlsx`).
* **Multi-PDF Knowledge Base:** Dynamically upload multiple PDF manuals, policy sheets, or FAQs through the sidebar UI to instantly update the local vector index.
* **Secure Configuration:** Loads API keys securely from a local `.env` file without exposing sensitive credentials in the web UI.

---

## 📂 Project Structure

```text
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python package dependencies
├── .env                    # Environment variables (API keys)
├── uploaded_pdfs/          # Directory containing uploaded PDF documents
├── faiss_index/            # Local FAISS vector database directory
└── Enquiry.xlsx       # Automated Excel interaction logs