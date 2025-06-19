# Role-based PDF QA Chatbot (LangChain + FAISS + Qwen2.5 via Ollama)

An intelligent chatbot powered by a local LLM that can role-play and answer questions based on PDF content.

## Technologies Used
- Python 3.10+
- Streamlit – Interface
- PyMuPDF (fitz) – PDF text extraction
- LangChain – Chunking, Retriever, PromptTemplate
- FAISS – Vector database
- Ollama (Qwen2.5:latest) – Embedding and LLM

## Project Folder Structure
```bash
pdf-chatbot/
│
├── app.py                  # Main application (Streamlit interface)
├── pdf_handler.py          # Extracts text from PDF and chunks it
├── embedder.py             # Embedding + FAISS database operations
├── chatbot.py              # Response generation with Qwen2.5 (via Ollama)
├── prompts.py              # Role-based prompt templates
├── roles.json              # Defines the list of roles
├── vectordb/               # FAISS files are stored here (created when the app runs)
├── data/                   # User-uploaded PDFs (created when the app runs)
├── requirements.txt        # Required libraries
└── README.md               # Project description
```

## Installation Instructions

1.  **Set up the Python environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Ensure Ollama is running:**
    Make sure the `qwen2.5:latest` model is running on Ollama. If not installed:
    ```bash
    ollama pull qwen2.5:latest
    ```
    The Ollama service needs to be running in the background (usually started with the `ollama serve` command or the Ollama Desktop application is running). It is not necessary to run the model separately with `ollama run qwen2.5:latest` while the application is running.

    To list installed models:
    ```bash
    ollama list
    ```
    You should see the `qwen2.5:latest` model (or a similar Qwen2.5 tag) in this list.


3.  **Start the application:**
    While in the project's main directory (`pdf-chatbot/`):
    ```bash
    streamlit run app.py
    ```

## Project Niche and Added Value

| Aspect          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| 📌 **Niche**    | An AI capable of role-playing and providing document-based consultancy.     |
| 🧠 **LLM Usage**| Local model (Qwen2.5 via Ollama) providing AI privacy and offline operation advantages. |
| 🛠️ **RAG Arch.**| Data-driven approach using LangChain + FAISS, differing from classic chatbots. |
| 💼 **CV Contrib.**| Combination of LangChain, FAISS, Ollama, PDF parsing, real-world use case. |
