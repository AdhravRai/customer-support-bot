import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.workflow import GraphWorkflow
import uuid

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


@st.cache_resource
def load_graph():
    workflow = GraphWorkflow()
    return workflow.build()


st.set_page_config(
    page_title="GigaCorp Customer Support",
    page_icon="🤖",
    layout="centered",
)

graph = load_graph()

st.title("🤖 GigaCorp Customer Support Assistant")

st.caption(
    "Ask questions about GigaCorp's shipping, returns, business hours, and service plans."
)

with st.sidebar:

    st.header("📋 About")

    st.write(
        """
This AI assistant answers customer support questions using
Retrieval-Augmented Generation (RAG).

### Tech Stack

- LangGraph
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM
- Streamlit
"""
    )

    st.subheader("💡 Try asking")

    st.markdown(
        """
- What is your refund policy?
- Do you ship internationally?
- What are your business hours?
- How do I return a product?
- What service plans do you offer?
"""
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):
            st.markdown("#### 📄 Sources")

            for source in message["sources"]:
                st.markdown(f"- {source}")

user_input = st.chat_input(
    "Ask a question about GigaCorp..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    chat_history = []

    for message in st.session_state.messages:

        if message["role"] == "user":
            chat_history.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        else:
            chat_history.append(
                AIMessage(
                    content=message["content"]
                )
            )

    state = {
        "question": user_input,
        "context": [],
        "answer": "",
        "sources": [],
        "chat_history": chat_history,
    }

    config = {
        "configurable": {
        "thread_id": st.session_state.thread_id
        }
    }

    try:

        with st.spinner("Searching knowledge base..."):

            result = graph.invoke(
                state,
                config=config,
            )

        answer = result["answer"]
        sources = result.get("sources", [])
        documents = result["context"]

        with st.expander("📚 Retrieved Context"):

            for i, doc in enumerate(documents, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content)
        
        with st.chat_message("assistant"):

            st.markdown(answer)

            if sources:

                st.markdown("#### 📄 Sources")

                for source in sources:
                    st.markdown(f"- {source}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

    except Exception as e:

        st.error("Something went wrong while generating the response.")

        st.exception(e)