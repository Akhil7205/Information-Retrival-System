import os
import sys
from rag_backend import load_llm, build_vectorstore, create_rag_chain
from langchain_core.documents import Document

def test_backend():
    print("Testing LLM loading...")
    try:
        llm = load_llm()
        print("LLM loaded successfully.")
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        return

    print("\nTesting Vectorstore creation...")
    try:
        # Create mock documents
        docs = [
            Document(page_content="This is a test document about AI.", metadata={"source": "test.pdf"}),
            Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "test.pdf"})
        ]
        vectorstore = build_vectorstore(docs)
        print("Vectorstore created successfully.")
    except Exception as e:
        print(f"Failed to create vectorstore: {e}")
        return

    print("\nTesting RAG Chain creation...")
    try:
        rag_chain = create_rag_chain(vectorstore)
        print("RAG Chain created successfully.")
    except Exception as e:
        print(f"Failed to create RAG chain: {e}")
        return

    print("\nBackend verification complete!")

if __name__ == "__main__":
    test_backend()
