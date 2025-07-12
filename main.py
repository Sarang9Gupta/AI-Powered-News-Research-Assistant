import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

# LangChain primitives
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq LLM (Llama 3 via Groq)
from langchain_groq.chat_models import ChatGroq

# Load env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

st.title("📰 News Research Tool")
st.sidebar.title("Enter News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_groq.pkl"

main_placeholder = st.empty()

# Groq-hosted Llama 3
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.9,
    max_tokens=500,
)

# --- Helper function for robust URL loading ---
def load_urls_robustly(urls_list, placeholder):
    loaded_docs = []
    failed_urls_info = []

    for i, url in enumerate(urls_list):
        if not url.strip(): # Skip empty strings
            continue
        try:
            placeholder.text(f"Attempting to load: {url}...")
            temp_loader = UnstructuredURLLoader(urls=[url])
            docs_from_url = temp_loader.load()
            loaded_docs.extend(docs_from_url)
            placeholder.text(f"✅ Successfully loaded: {url}")
        except Exception as e:
            failed_urls_info.append(f"URL {i+1} ({url}): {e}")
            placeholder.warning(f"❌ Failed to load URL {i+1} ({url}): {e}")
            
    return loaded_docs, failed_urls_info
# --- End of helper function ---


if process_url_clicked:
    main_placeholder.text("🔄 Loading URLs...")
    
    # Filter out empty URLs before passing to helper
    valid_urls = [url.strip() for url in urls if url.strip()]
    
    if not valid_urls:
        main_placeholder.warning("⚠️ No URLs entered. Please provide at least one URL.")
        st.stop()

    all_loaded_docs, failed_urls_summary = load_urls_robustly(valid_urls, main_placeholder)
    
    if not all_loaded_docs:
        main_placeholder.error("🔴 No documents could be loaded from any of the provided URLs. Please check them.")
        st.stop()

    main_placeholder.text("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = splitter.split_documents(all_loaded_docs)

    main_placeholder.text("🔍 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("💾 Saving index to disk...")
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    if failed_urls_summary:
        st.warning("⚠️ Some URLs could not be processed:")
        for info in failed_urls_summary:
            st.warning(f"- {info}")
    
    main_placeholder.success("✅ URLs processed. You can now ask questions!")

query = main_placeholder.text_input("🔎 Ask a question:")
if query:
    if not os.path.exists(file_path):
        st.warning("⚠️ Please process URLs first.")
    else:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.header("🧠 Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("🔗 Sources")
            for src in sources.split("\n"):
                st.markdown(f"- {src}")