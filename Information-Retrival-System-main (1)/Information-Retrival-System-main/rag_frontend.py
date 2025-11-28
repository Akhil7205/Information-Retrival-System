import streamlit as st
from rag_backend import (
    load_and_split_pdfs,
    build_vectorstore,
    create_rag_chain,
)

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 Chat with your PDFs")


# ---------------------------
# Session State
# ---------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------
# Sidebar — PDF Upload
# ---------------------------
with st.sidebar:
    st.header("📂 Upload PDFs")

    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"]
    )

    if st.button("Process PDFs"):
        if not pdf_docs:
            st.warning("Upload at least one PDF file.")
        else:
            with st.spinner("Processing your documents..."):
                splits = load_and_split_pdfs(pdf_docs)
                vectorstore = build_vectorstore(splits)
                st.session_state.rag_chain = create_rag_chain(vectorstore)

            st.success("PDFs processed! Start chatting.")


# ---------------------------
# Chat UI
# ---------------------------
if st.session_state.rag_chain:
    user_message = st.chat_input("Ask anything from your PDF...")

    if user_message:
        st.session_state.chat_history.append(("user", user_message))

        response = st.session_state.rag_chain.invoke(user_message)
        st.session_state.chat_history.append(("assistant", response))

    # display messages
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

else:
    st.info("👈 Upload and process PDFs to begin.")
