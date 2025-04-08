import streamlit as st
import fitz
import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

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

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Smart Q&A Assistant", layout="wide")
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.markdown("<h1 style='text-align:center;'>Smart chat assistance</h1>", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_embedding(table, chunk, embedding, source):
    try:
        clear_table(table)
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
        supabase.table(table_name).delete().neq('chunk',"").execute()
        # st.success(f"Cleared table: {table_name}")
    except Exception as e:
        pass
        # st.error(f"Failed to clear table {table_name}: {e}")

def scrape_website(url, max_pages=10, max_depth=2):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    scraped_text = []
    visited = set()
    to_visit = [(url, 0)]

    while to_visit and len(visited) < max_pages:
        current_url, depth = to_visit.pop(0)
        if current_url in visited or depth > max_depth:
            continue

        try:
            driver.get(current_url)
            time.sleep(1.5)
            visited.add(current_url)

            content = []
            for tag in ['h1', 'h2', 'h3', 'p', 'li']:
                elements = driver.find_elements(By.TAG_NAME, tag)
                content.extend([e.text.strip() for e in elements if e.text.strip()])
            page_text = "\n".join(content)
            if page_text:
                scraped_text.append((page_text, current_url))

            for link in driver.find_elements(By.TAG_NAME, "a"):
                href = link.get_attribute("href")
                if href and href.startswith(url) and href not in visited:
                    to_visit.append((href, depth + 1))
        except Exception as e:
            st.warning(f"Failed to fetch {current_url}: {e}")

    driver.quit()
    return scraped_text

# Initialize QA history
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Select tab (source)
tab_choice = st.radio("Select Source", ["Upload PDF", "Scrape Website"], horizontal=True)
clear_table("chat_history")
if tab_choice == "Upload PDF":
    st.session_state.active_tab = "pdf"
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploaded_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            chunks = extract_text_from_pdf(file_path)
            embeddings = [embed_model.encode(chunk).tolist() for chunk in chunks]
            for chunk, embedding in zip(chunks, embeddings):
                store_embedding("pdf_embeddings", chunk, embedding, uploaded_file.name)
        st.success("PDFs processed.")

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
            st.success("Scraped content processed.")

# Sidebar for chat history
with st.sidebar:
    st.markdown("### Chat History:")
    for i, qa in enumerate(st.session_state.qa_history):
        with st.expander(f"Q{i+1}: {qa['question']}"):
            st.markdown(f"**A{i+1}:** {qa['answer']}")

# Main Q&A Section
st.markdown("### Ask a question based on uploaded PDFs or scraped website:")
user_question = st.text_input("Ask your question here")

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
        prompt = f"Use the following context to answer the question:\n\n{user_question}\n\nContext:\n{context}"

        try:
            response = model.generate_content(prompt)
            final_answer = response.text if hasattr(response, "text") else "No answer found."
        except Exception as e:
            final_answer = f"Error: {e}"

        st.markdown(f"**Answer:** {final_answer}")
        st.session_state.qa_history.append({"question": user_question, "answer": final_answer})

        try:
            supabase.table("chat_history").insert({
                "question": user_question,
                "response": final_answer,
                "source": source
            }).execute()
        except Exception as e:
            st.warning(f"Failed to save chat: {e}")
