import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']

    for i, message in enumerate(st.session_state.chatHistory):
        role = "user" if i % 2 == 0 else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)


def main():
    st.set_page_config(page_title="Information Retrieval System 💁", layout="wide")
    st.title("📄 Chat with Pdf.")

    # Initialize session state if not exists
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📂 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click 'Process'", accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("⏳ Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("✅ Processing Complete!")

    # Chat interface
    if st.session_state.conversation:
        user_question = st.chat_input("Ask a question about your PDFs...")
        if user_question:
            handle_user_input(user_question)
    else:
        st.info("👈 Upload and process PDFs to start chatting.")


if __name__ == "__main__":
    main()
