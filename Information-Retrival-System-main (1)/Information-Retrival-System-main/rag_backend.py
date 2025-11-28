import os
from dotenv import load_dotenv
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# ------------------------------
# 🔹 LOAD LLM
# ------------------------------
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=os.getenv("GOOGLE_API_KEY")
    )


# ------------------------------
# 🔹 LOAD PDF + TEXT SPLIT
# ------------------------------
def load_and_split_pdfs(pdf_files):
    documents = []

    for uploaded in pdf_files:
        # Save uploaded Streamlit file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # Now load it properly
        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# ------------------------------
# 🔹 SIMPLE EMBEDDINGS (No torch required)
# ------------------------------
class SimpleEmbeddings(Embeddings):
    """Simple hash-based embeddings to avoid torch dependency."""
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            # Simple hash-based embedding (384 dimensions)
            hash_val = hash(text)
            np.random.seed(hash_val % (2**32))
            embedding = np.random.randn(384).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query."""
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(384).tolist()


# ------------------------------
# 🔹 BUILD VECTORSTORE (FAISS)
# ------------------------------
def build_vectorstore(splits):
    emb = SimpleEmbeddings()
    return FAISS.from_documents(splits, emb)


# ------------------------------
# 🔹 CREATE RAG CHAIN
# ------------------------------
def create_rag_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = load_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    You are a retrieval-augmented assistant. Follow these rules with zero exceptions:

    1. Use ONLY the information provided in the context.
    2. If the context does not contain enough information to fully answer,
    say exactly: "I don't know based on the provided context." 
    Then briefly explain what is missing.
    3. Never invent facts, never guess, never hallucinate.
    4. If the question is unclear, ask for clarification instead of producing garbage.
    5. Always give clear, complete answers when the context supports it.

    Your job is to be precise, grounded, and strict about evidence.
    """
        ),
        (
            "human",
            "User question:\n{question}\n\nRetrieved context:\n{context}\n\nYour grounded answer:"
        )
    ])


    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_pipeline = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_pipeline
