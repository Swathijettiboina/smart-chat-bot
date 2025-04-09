import streamlit as st
import fitz
import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
import google.generativeai as genai
import time
import torch
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Load environment variables
load_dotenv()
nest_asyncio.apply()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
API_KEY = os.getenv("API_KEY", "") 

if not all([SUPABASE_URL, SUPABASE_KEY, API_KEY]):
    st.error("Missing environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model = embed_model.to(device)  # after full loading

st.set_page_config(page_title="Smart Q&A Assistant", layout="wide")
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.markdown("<h1 style='text-align:center;'>Smart Chat Assistance</h1>", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    # Save full content for verification
    with open(pdf_path.replace(".pdf", ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_embedding(table, chunk, embedding, source):
    try:
        supabase.table(table).insert({
            "chunk": chunk,
            "embedding": embedding,
            "source": source
        }).execute()
    except Exception as e:
        st.error(f"Error storing in {table}: {e}")

def search_embeddings(rpc_func, query_embedding, top_k=10):
    try:
        response = supabase.rpc(rpc_func, {
            "query": query_embedding,
            "match_count": top_k
        }).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error searching {rpc_func}: {e}")
        return []

def clear_table(table_name):
    try:
        supabase.table(table_name).delete().neq('chunk', "").execute()
    except Exception:
        pass
def scrape_website(url, depth=5, visited=None):
    if visited is None:
        visited = set()
    if depth == 0 or url in visited:
        return []   
    visited.add(url)
    scraped_data = []
   
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            page_text = " ".join([p.get_text() for p in paragraphs])
            if page_text.strip():
                scraped_data.append((page_text, url))
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    scraped_data.extend(scrape_website(full_url, depth-1, visited))
    except Exception as e:
        pass

    # Save scraped content for verification
    with open("scraped_website.txt", "a", encoding="utf-8") as f:
        for text, src in scraped_data:
            f.write(f"\n\n=== Source: {src} ===\n{text}\n")

    return scraped_data

# Initialize session state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "initialized" not in st.session_state:
    clear_table("pdf_embeddings")
    clear_table("web_embeddings")
    clear_table("chat_history")
    st.session_state.initialized = True

# Tab selection
tab_choice = st.radio("Select Source", ["Upload PDF", "Scrape Website"], horizontal=True)

if tab_choice == "Upload PDF":
    st.session_state.active_tab = "pdf"
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            chunks = extract_text_from_pdf(file_path)
            embeddings = [embed_model.encode(chunk).tolist() for chunk in chunks]
            for chunk, embedding in zip(chunks, embeddings):
                store_embedding("pdf_embeddings", chunk, embedding, uploaded_file.name)
        st.success("PDFs processed and stored.")

elif tab_choice == "Scrape Website":
    st.session_state.active_tab = "web"
    st.subheader("Scrape Website")
    scrape_url = st.text_input("Website URL", placeholder="https://example.com")
    if st.button("Scrape Website"):
        if scrape_url:
            with st.spinner("Scraping..."):
                scraped_data = scrape_website(scrape_url)
                embeddings = [embed_model.encode(text).tolist() for text, _ in scraped_data]
                for (chunk, source), embedding in zip(scraped_data, embeddings):
                    store_embedding("web_embeddings", chunk, embedding, source)
            st.success("Website content scraped and stored.")

# Sidebar - History
with st.sidebar:
    st.markdown("### Chat History")
    for i, qa in enumerate(st.session_state.qa_history):
        with st.expander(f"Q{i+1}: {qa['question']}"):
            st.markdown(f"**Answer:** {qa['answer']}")

# Main Q&A Interface
st.markdown("### Ask a question based on uploaded PDFs or scraped website:")
user_question = st.text_input("Ask your question here", key="question_input")

if st.button("Get Answer"):
    if user_question:
        query_embedding = embed_model.encode(user_question).tolist()
        source = st.session_state.active_tab

        if source == "pdf":
            results = search_embeddings("match_pdf_embeddings", query_embedding, top_k=10)
        elif source == "web":
            results = search_embeddings("match_web_embeddings", query_embedding, top_k=10)
        else:
            st.warning("Please upload a PDF or scrape a website first.")
            results = []

        context = "\n".join(row['chunk'] for row in results)
        prompt = f"Use the following context to answer the question:\n\nQuestion: {user_question}\n\nContext:\n{context}"

        try:
            response = model.generate_content(prompt)
            final_answer = response.text if hasattr(response, "text") else "No answer found."
        except Exception as e:
            final_answer = f"Error: {e}"

        st.markdown(f"**Question:** {user_question}")
        st.markdown(f"**Answer:** {final_answer}")

        # Save in history
        st.session_state.qa_history.append({
            "question": user_question,
            "answer": final_answer
        })

        # Save in DB
        try:
            supabase.table("chat_history").insert({
                "question": user_question,
                "response": final_answer,
                "source": source
            }).execute()
        except Exception as e:
            st.warning(f"Failed to save chat: {e}")

        # Clear the input
        # st.experimental_rerun()
