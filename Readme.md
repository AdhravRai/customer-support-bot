# 🤖 GigaCorp Customer Support Assistant

An AI-powered Customer Support Assistant built using **LangGraph**, **LangChain**, **FAISS**, and **Streamlit**. The application uses Retrieval-Augmented Generation (RAG) to answer customer queries from a local knowledge base while maintaining conversational context across multiple interactions.

---

## 📌 Project Overview

This project simulates a customer support chatbot for a fictional company called **GigaCorp**.

Instead of relying solely on a Large Language Model, the chatbot retrieves relevant information from a local FAQ document using a FAISS vector database. The retrieved context is then provided to the LLM to generate accurate and context-aware responses.

The application also supports conversational memory, allowing users to ask follow-up questions naturally.

---

## ✨ Features

- 🔍 Retrieval-Augmented Generation (RAG)
- 📄 Local FAQ knowledge base
- ⚡ FAISS vector store for semantic search
- 🤖 LangGraph workflow orchestration
- 🧠 Conversational memory for follow-up questions
- 📚 Source citations for retrieved answers
- 💬 Interactive Streamlit chat interface
- 🚀 Fast inference using Groq LLM
- 🧩 HuggingFace sentence embeddings

---

## 🏗️ Project Architecture

```
                    User
                      │
                      ▼
            Streamlit Chat Interface
                      │
                      ▼
              LangGraph Workflow
          ┌───────────┴───────────┐
          ▼                       ▼
   Retrieve Node           Generate Node
          │                       │
          ▼                       ▼
     FAISS Retriever        Groq LLM
          │
          ▼
 HuggingFace Embeddings
          │
          ▼
 GigaCorp FAQ Knowledge Base
```

---

## 📂 Project Structure

```text
customer-support-rag/
│
├── data/
│   └── documents/
│       └── gigacorp_faq.txt
│
├── src/
│   ├── config/
│   ├── embeddings/
│   ├── graph/
│   ├── ingestion/
│   ├── prompts/
│   ├── utils/
│   └── vector_store/
│
├── tests/
│
├── build_index.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .env.example
```

---

## ⚙️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| LLM | Groq |
| Framework | LangChain |
| Workflow | LangGraph |
| Vector Store | FAISS |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| UI | Streamlit |
| Document Loader | PyPDF, TextLoader |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/customer-support-rag.git

cd customer-support-rag
```

---

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure environment variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

### 5. Build the Vector Store

```bash
python build_index.py
```

This command

- Loads the FAQ document
- Splits it into chunks
- Generates embeddings
- Creates a FAISS vector database

---

### 6. Run the application

```bash
streamlit run streamlit_app.py
```

---

## 💬 Example Questions

Try asking:

- What are your business hours?
- What is your refund policy?
- Do you ship internationally?
- How can I return a product?
- What service plans do you offer?
- Do you ship to India?
- How much does it cost there?

---

## 🧠 How the RAG Pipeline Works

1. User submits a question.
2. The question is embedded using HuggingFace embeddings.
3. FAISS retrieves the most relevant document chunks.
4. Retrieved context and conversation history are passed to the LangGraph workflow.
5. Groq generates a response using only the retrieved context.
6. The answer and source document are displayed in the Streamlit interface.

---

## 📸 Screenshots

Add screenshots of the following after deployment:

- Chat Interface
- Source Citations
- Retrieved Context
- Sidebar
- Example Conversation

---

## 🔮 Future Improvements

- Support multiple PDF documents
- Add streaming responses
- Store conversations in a database
- User authentication
- Feedback and rating system
- Hybrid search (keyword + semantic retrieval)

---

## 👨‍💻 Author

**Adhrav Rai**

B.Tech Computer Science Student

Built as part of an AI Engineering assignment to demonstrate Retrieval-Augmented Generation, LangGraph workflows, conversational memory, and LLM-powered customer support.

---
## 📸 Application Preview

### Home Screen

![Home](assets/screenshots/home_page.png)

The chatbot provides an intuitive chat interface for asking customer support questions.

---

### Conversational Memory & Source Citation

![Conversation](assets/screenshots/conversation.png)

The assistant remembers previous messages, retrieves relevant document chunks using FAISS, and cites the source document used to answer the query.