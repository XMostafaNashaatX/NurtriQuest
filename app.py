import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="NutriQuest",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "NutriQuest - A quest for the best nutrition and workout plans"
    }
)

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import time
import logging
import requests
import os
import tarfile
import urllib.request
from pathlib import Path
import json
import traceback
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
import dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    dotenv.load_dotenv()
else:
    # Create .env file with default values if it doesn't exist
    with open(".env", "w") as env_file:
        env_file.write(f"""# Hugging Face API Key - Replace with your actual key
HF_API_TOKEN=YOUR_HUGGING_FACE_TOKEN_HERE

# Google API Key
GOOGLE_API_KEY=
GOOGLE_CSE_ID=

# Admin settings
ADMIN_EMAIL=s-mostafa.abdelhameed@zewailcity.edu.eg
ADMIN_PASSWORD=

# Email sending settings (for verification codes)
EMAIL_SENDER=nutriquest.app@gmail.com
EMAIL_PASSWORD=
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
""")
    dotenv.load_dotenv()

# Get API keys from environment variables
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Default to empty string if not set
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "s-mostafa.abdelhameed@zewailcity.edu.eg")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")  # Default password removed for security

# Email settings
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "nutriquest.app@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# Add .env to .gitignore to prevent API keys from being committed
if not os.path.exists(".gitignore"):
    with open(".gitignore", "w") as gitignore_file:
        gitignore_file.write(".env\n")
elif ".env" not in open(".gitignore").read():
    with open(".gitignore", "a") as gitignore_file:
        gitignore_file.write("\n.env\n")

# Configure logging
logging.getLogger("org.terrier").setLevel(logging.ERROR)

# Admin authentication helpers (simplified to use password)
def verify_admin_password(password):
    """Check if the password matches the admin password"""
    return password == ADMIN_PASSWORD

# Session state initialization
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "custom_models" not in st.session_state:
    st.session_state.custom_models = []
if "default_model" not in st.session_state:
    st.session_state.default_model = "meta-llama/Llama-2-70b-chat-hf"
if "search_index" not in st.session_state:
    st.session_state.search_index = None
if "unique_terms" not in st.session_state:
    st.session_state.unique_terms = []
if "inverted_index" not in st.session_state:
    st.session_state.inverted_index = {}

# Function to check if environment variables are properly set
def check_environment_variables():
    """Check if required environment variables are set and guide the user if not"""
    missing_vars = []
    
    if not HF_API_TOKEN:
        missing_vars.append("HF_API_TOKEN")
    
    if not ADMIN_PASSWORD:
        missing_vars.append("ADMIN_PASSWORD")
    
    if missing_vars:
        st.error("Missing environment variables detected!")
        st.warning(f"Please set the following in your .env file: {', '.join(missing_vars)}")
        
        with st.expander("How to set up your environment"):
            st.markdown("""
            ### Setting up your environment
            
            1. Create a file named `.env` in the same directory as this script
            2. Add the following content to the file:
            ```
            HF_API_TOKEN=your_hugging_face_api_key
            ADMIN_PASSWORD=your_admin_password
            ```
            3. Replace the values with your actual API key and password
            4. Restart the application
            
            Note: The `.env` file is not shared when you push to GitHub.
            """)
        return False
    return True

@st.cache_resource
def ensure_java():
    """Download and unpack OpenJDK 11 if not already present."""
    java_dir = Path.home() / ".java"
    if not java_dir.exists():
        java_dir.mkdir(parents=True)
        # Example JDK 11 URL; you may choose a more recent build if desired
        jdk_url = (
            "https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz"
        )
        archive = java_dir / "jdk11.tar.gz"
        urllib.request.urlretrieve(jdk_url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(java_dir)
    # Locate the unpacked JDK root
    jdk_root = next(java_dir.glob("jdk-*"))
    os.environ["JAVA_HOME"] = str(jdk_root)
    os.environ["PATH"] = f"{jdk_root}/bin:" + os.environ.get("PATH", "")
    return True

# Ensure Java is in place before any PyTerrier initialization
ensure_java()

# Initialize PyTerrier after Java setup
import pyterrier as pt
if not pt.java.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('words')

download_nltk_resources()

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stem_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def remove_extra_spaces(text):
    words = text.split()
    return " ".join(words)

def clean(sentence):
    tokens = word_tokenize(sentence)
    cleaned = []
    for text in tokens:
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"RT ", " ", text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub(r"@[\w]*", " ", text)
        text = re.sub(r"[\.\,\#_\|\:\?\?!&\-$;'/\=]", " ", text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r"\s+", " ", text)
        cleaned.append(text)
    return ' '.join(cleaned)

def preprocess_document(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r"RT ", " ", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"@[\w]*", " ", text)
    text = re.sub(r"[\.\,\#\(\)_\|\:\?\?!&\-\$;'/\=]", " ", text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\s+", " ", text)

    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return ' '.join(stemmed_words)

def preprocess_query(query):
    query = re.sub(r"http\S+", " ", query)
    query = re.sub(r'\([^)]*\)', '', query)
    query = re.sub(r"RT ", " ", query)
    query = re.sub('[^a-zA-Z]', ' ', query)
    query = re.sub(r"@[\w]*", " ", query)
    query = re.sub(r"[\.\,\#\(\)_\|\:\?\?!&\-\$;'/\=]", " ", query)
    query = re.sub(r'\t', ' ', query)
    query = re.sub(r'\n', ' ', query)
    query = re.sub(r"\s+", " ", query)

    words = word_tokenize(query.lower())
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return ' '.join(stemmed_words)

def retrieve_documents(index, query, lexicon, inverted_index, top_n=10):
    tokenized = word_tokenize(query)
    documents = []

    for word in tokenized:
        if word in lexicon:
            for posting in inverted_index.get(word, []):
                doc_id = posting[0]
                documents.append(doc_id)

    return list(set(documents))[:top_n]

def tf_idf_ranking(index, query, lexicon, inverted_index, df, top_n=10):
    tfidf_retr = pt.terrier.Retriever(index, wmodel="TF_IDF", num_results=top_n)
    results = tfidf_retr.search(query)

    ranked_docs = []
    for i in range(len(results)):
        doc_id = results["docid"][i]
        ranked_docs.append(doc_id)

    return ranked_docs

def bm25_ranking(index, query, top_n=10):
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=top_n)
    results = bm25.search(query)

    ranked_docs = []
    for i in range(len(results)):
        doc_id = results["docid"][i]
        ranked_docs.append(doc_id)

    return ranked_docs

def query_expansion_rm3(index, query, df, top_n=10):
    try:
        bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=20)
        bm25_results = bm25.search(query)

        if len(bm25_results) == 0:
            return query, []

        expansion_terms = {}

        # Track sources used for expansion to ensure diversity
        sources_used = {
            'Wikipedia': 0,
            'YouTube': 0,
            'Google': 0
        }
        max_per_source = 2  # Maximum documents to use from each source

        for i in range(min(10, len(bm25_results))):
            doc_id = int(bm25_results["docid"][i])
            if doc_id < 0 or doc_id >= len(df):
                continue
                
            # Get source information
            source = df.iloc[doc_id]['source']
            
            # Skip if we've already used enough from this source
            if sources_used.get(source, 0) >= max_per_source:
                continue
                
            # Increment source counter
            sources_used[source] = sources_used.get(source, 0) + 1
            
            # Get text based on source type
            if source == 'Wikipedia':
                # For Wikipedia, use the wiki_content
                doc_text = str(df.iloc[doc_id].get('wiki_content', ''))
            else:
                # For other sources, use the topic and derived terms from the link
                topic = str(df.iloc[doc_id]['topic'])
                link = str(df.iloc[doc_id]['link'])
                
                # Extract terms from link (e.g., from YouTube search URLs or Google search URLs)
                link_terms = ''
                if 'youtube.com' in link and 'search_query=' in link:
                    # Extract search terms from YouTube link
                    search_part = link.split('search_query=')[-1].split('&')[0]
                    link_terms = search_part.replace('+', ' ').replace('%20', ' ')
                elif 'google.com/search' in link and 'q=' in link:
                    # Extract search terms from Google link
                    search_part = link.split('q=')[-1].split('&')[0]
                    link_terms = search_part.replace('+', ' ').replace('%20', ' ')
                
                doc_text = f"{topic} {link_terms}"
                
            if not isinstance(doc_text, str) or not doc_text.strip():
                continue

            words = word_tokenize(doc_text.lower())
            filtered_words = [w for w in words if w not in stop_words and len(w) > 3]

            for word in filtered_words:
                if word in expansion_terms:
                    expansion_terms[word] += 1
                else:
                    expansion_terms[word] = 1

        sorted_terms = sorted(expansion_terms.items(), key=lambda x: x[1], reverse=True)
        top_terms = [term for term, count in sorted_terms[:3]]

        if top_terms:
            expanded_query = query + " " + " ".join(top_terms)

            results = bm25.search(expanded_query)

            ranked_docs = []
            for i in range(len(results)):
                doc_id = results["docid"][i]
                ranked_docs.append(doc_id)

            return expanded_query, ranked_docs
        else:
            return query, bm25_ranking(index, query, top_n)

    except Exception as e:
        st.warning(f"Query expansion error: {e}")
        return query, bm25_ranking(index, query, top_n)

def calculate_precision_recall(relevant_docs, retrieved_docs):
    try:
        relevant_set = set(int(doc_id) for doc_id in relevant_docs)
        retrieved_set = set(int(doc_id) for doc_id in retrieved_docs)

        relevant_retrieved = relevant_set.intersection(retrieved_set)

        precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0

        return precision, recall
    except Exception as e:
        st.warning(f"Error in precision/recall calculation: {e}")
        return 0, 0

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate_ndcg(relevant_docs, retrieved_docs):
    try:
        relevant_set = set(int(doc_id) for doc_id in relevant_docs)
        retrieved_int = [int(doc_id) for doc_id in retrieved_docs]

        dcg = 0
        for i, doc_id in enumerate(retrieved_int):
            if doc_id in relevant_set:
                dcg += 1 / np.log2(i + 2)

        idcg = 0
        for i in range(min(len(relevant_set), len(retrieved_docs))):
            idcg += 1 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0

        return ndcg
    except Exception as e:
        st.warning(f"Error in NDCG calculation: {e}")
        return 0

def extract_keywords_from_link(link):
    """Extract meaningful keywords from YouTube or Google links."""
    try:
        keywords = ""
        if 'youtube.com' in link and 'search_query=' in link:
            # Extract search terms from YouTube link
            search_part = link.split('search_query=')[-1].split('&')[0]
            keywords = search_part.replace('+', ' ').replace('%20', ' ')
        elif 'google.com/search' in link and 'q=' in link:
            # Extract search terms from Google link
            search_part = link.split('q=')[-1].split('&')[0]
            keywords = search_part.replace('+', ' ').replace('%20', ' ')
        elif 'wikipedia.org/wiki/' in link:
            # Extract topic from Wikipedia link
            topic = link.split('/wiki/')[-1].replace('_', ' ')
            keywords = topic
        
        # Clean up keywords
        keywords = re.sub(r'[^a-zA-Z0-9\s]', ' ', keywords)
        keywords = re.sub(r'\s+', ' ', keywords).strip()
        
        if keywords:
            return f"Keywords: {keywords}"
        else:
            # Fallback to just returning the domain
            domain = link.split('//')[1].split('/')[0]
            return f"Source: {domain}"
    except Exception as e:
        print(f"Error extracting keywords from link: {e}")
        return ""

def comprehensive_search(index, query, df, unique_terms, inverted_index, top_n=10, use_query_expansion=True, use_bert=False, selected_sources=None):
    start_time = time.time()

    processed_query = preprocess_query(query)

    st.markdown(f"**Processed Query:** {processed_query}")

    source_specific_query = processed_query
    if selected_sources and len(selected_sources) < len(df['source'].unique()):
        source_terms = " ".join(selected_sources)
        source_specific_query = f"{processed_query} {source_terms}"
        st.markdown(f"**Source-boosted Query:** {source_specific_query}")

    if use_query_expansion:
        expanded_query, expanded_docs = query_expansion_rm3(index, source_specific_query, df, top_n=top_n*5)
        st.markdown(f"**Expanded Query:** {expanded_query}")
        docs_to_filter = expanded_docs
    else:
        expanded_query = processed_query
        docs_to_filter = bm25_ranking(index, source_specific_query, top_n=top_n*5)

    doc_details = []
    for doc_id in docs_to_filter:
        doc_idx = int(doc_id)
        if doc_idx >= len(df):
            continue

        try:
            source = df.iloc[doc_idx]['source']

            if selected_sources and source not in selected_sources:
                continue

            content = str(df.iloc[doc_idx]['content']).lower() if 'content' in df.columns else ""
            topic = str(df.iloc[doc_idx]['topic']).lower()
            
            # Add source-specific content handling
            if source == 'Wikipedia' and 'wiki_content' in df.columns:
                wiki_content = str(df.iloc[doc_idx]['wiki_content']).lower()
                if isinstance(wiki_content, str):
                    content += " " + wiki_content

            query_terms = set(processed_query.lower().split())
            content_matches = sum(1 for term in query_terms if term in content)
            topic_matches = sum(1 for term in query_terms if term in topic)
            
            # Give different weights to different sources
            source_weight = 1.0
            if source == 'Wikipedia':
                source_weight = 1.0  # Full weight
            elif source == 'YouTube':
                source_weight = 0.9  # Slightly less
            elif source == 'Google':
                source_weight = 0.8  # Even less
                
            relevance_score = ((content_matches * 0.3) + (topic_matches * 0.7)) * source_weight

            doc_details.append({
                'id': doc_id,
                'idx': doc_idx,
                'topic': df.iloc[doc_idx]['topic'],
                'source': source,
                'relevance_score': relevance_score,
                'content': content[:300]
            })
        except Exception as e:
            print(f"Error processing doc_id {doc_id}: {e}")
            continue

    doc_details.sort(key=lambda x: x['relevance_score'], reverse=True)

    docs_by_source = {}
    for doc in doc_details:
        source = doc['source']
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)

    final_docs = []
    sources = list(docs_by_source.keys())

    if not sources:
        return expanded_query, [], time.time() - start_time

    source_quotas = {}
    base_quota = top_n // len(sources) if sources else 0
    remainder = top_n - (base_quota * len(sources))

    for i, source in enumerate(sources):
        source_quotas[source] = base_quota + (1 if i < remainder else 0)

    for source, quota in source_quotas.items():
        source_docs = docs_by_source.get(source, [])
        source_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        final_docs.extend([doc['id'] for doc in source_docs[:quota]])

    if len(final_docs) < top_n:
        remaining_docs = [
            doc['id'] for doc in doc_details
            if doc['id'] not in final_docs
        ]
        final_docs.extend(remaining_docs[:top_n - len(final_docs)])

    end_time = time.time()
    search_time = end_time - start_time

    return expanded_query, final_docs, search_time

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bodybuilding_search_results.csv')
        
        # Preprocess wiki_content
        df['processed_wiki'] = df['wiki_content'].apply(preprocess_document)
        
        # Ensure content field exists and is populated for all sources
        if 'content' not in df.columns:
            # Create more meaningful content for all sources
            df['content'] = df.apply(lambda row: 
                # For Wikipedia, use wiki_content
                row['wiki_content'] if row['source'] == 'Wikipedia' else 
                # For YouTube, create enriched content from topic and link
                (f"Video about {row['topic']}. " + 
                extract_keywords_from_link(row['link']) if row['source'] == 'YouTube' else
                # For Google, create enriched content from topic and link
                f"Search result about {row['topic']}. " + 
                extract_keywords_from_link(row['link']))
                , axis=1)
        
        # Add a combined content field that integrates all available information
        df['combined_content'] = df.apply(lambda row: 
            f"{row['topic']} {row['source']} {row.get('content', '')} {row.get('wiki_content', '')}", 
            axis=1)
        
        # Process the combined content for searching
        df['processed'] = df['combined_content'].apply(preprocess_document)

        sources_count = df['source'].value_counts()
        print(f"Dataset sources distribution: {sources_count}")

        needs_augmentation = False

        if 'YouTube' not in df['source'].values or df[df['source'] == 'YouTube'].shape[0] < 10:
            needs_augmentation = True
            youtube_entries = df[df['source'] == 'Wikipedia'].copy().head(20)
            youtube_entries['source'] = 'YouTube'
            youtube_entries['link'] = youtube_entries['link'].apply(
                lambda x: f"https://www.youtube.com/results?search_query={x.split('/')[-1].replace('_', '+')}"
            )
            # Keep wiki_content but mark it as from YouTube adaptation
            youtube_entries['content'] = youtube_entries.apply(
                lambda row: f"Video about {row['topic']}. Keywords: {extract_keywords_from_link(row['link'])}", axis=1
            )
            df = pd.concat([df, youtube_entries], ignore_index=True)

        if 'Google' not in df['source'].values or df[df['source'] == 'Google'].shape[0] < 10:
            needs_augmentation = True
            google_entries = df[df['source'] == 'Wikipedia'].copy().iloc[20:40]
            google_entries['source'] = 'Google'
            google_entries['link'] = google_entries['link'].apply(
                lambda x: f"https://www.google.com/search?q={x.split('/')[-1].replace('_', '+')}"
            )
            # Keep wiki_content but mark it as from Google adaptation
            google_entries['content'] = google_entries.apply(
                lambda row: f"Search result about {row['topic']}. Keywords: {extract_keywords_from_link(row['link'])}", axis=1
            )
            df = pd.concat([df, google_entries], ignore_index=True)

        # Regenerate combined content after augmentation
        df['combined_content'] = df.apply(lambda row: 
            f"{row['topic']} {row['source']} {row.get('content', '')} {row.get('wiki_content', '')}", 
            axis=1)
        
        # Regenerate processed content after augmentation
        df['processed'] = df['combined_content'].apply(preprocess_document)

        print(f"After augmentation, sources distribution: {df['source'].value_counts()}")

        if needs_augmentation:
            df.to_csv('augmented_dataset.csv', index=False)
            print("Saved augmented dataset to augmented_dataset.csv")

        return df, needs_augmentation
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=['topic', 'source', 'link', 'wiki_content', 'processed', 'content']), False

@st.cache_resource(show_spinner=False)
def create_index(df):
    try:
        df['docno'] = df.index.astype(str)

        if df['processed'].isna().any():
            print("Warning: Some documents have no processed content. Adding minimal content...")
            df.loc[df['processed'].isna(), 'processed'] = df.loc[df['processed'].isna(), 'topic']

        # Create source-specific metrics
        source_topics = {}
        for source in df['source'].unique():
            topics = df[df['source'] == source]['topic'].unique().tolist()
            source_topics[source] = topics[:5]
        print(f"Topics by source: {source_topics}")

        print("Creating new index...")
        indexer = pt.DFIndexer("./DatasetIndex", overwrite=True)
        # Use the combined processed field that includes all sources
        index_ref = indexer.index(df["processed"], df["docno"])

        index = pt.IndexFactory.of(index_ref)
        print(f"Index created with {index.getCollectionStatistics().getNumberOfDocuments()} documents")

        unique_terms = []
        for kv in index.getLexicon():
            unique_terms.append(kv.getKey())
        print(f"Extracted {len(unique_terms)} unique terms")

        inverted_index = {}
        for kv in index.getLexicon():
            postings = []
            for j in index.getInvertedIndex().getPostings(index.getLexicon()[kv.key]):
                postings.append([j.getId(), j.getFrequency()])
            inverted_index[kv.key] = postings

        test_sources = {}
        retriever = pt.terrier.Retriever(index, wmodel="BM25", num_results=5)

        for source in df['source'].unique():
            results = retriever.search(source)
            if len(results) > 0:
                test_sources[source] = True
            else:
                test_sources[source] = False

        print(f"Source indexing test: {test_sources}")

        return index, unique_terms, inverted_index
    except Exception as e:
        st.error(f"Error creating index: {e}")
        traceback.print_exc()
        return None, [], {}

def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    .main {
        background-color: #121212;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffdd57;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .subtitle {
        color: #e0e0e0;
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 40px;
    }
    .search-box {
        background-color: #1e1e1e;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 30px;
        border: 1px solid #333;
    }
    .result-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        border-left: 5px solid #ffdd57;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .stButton button {
        background-color: #ffdd57;
        color: #121212;
        font-weight: 600;
        border-radius: 8px;
        padding: 5px 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #ffd633;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .stTextInput input, .stTextInput textarea {
        background-color: #2c2c2c;
        color: #e0e0e0;
        border: 1px solid #444;
    }
    .stTextInput label {
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #1e1e1e;
        border-right: 1px solid #333;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
    .stSidebar [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    .stSidebar [data-testid="stMarkdownContainer"] h1,
    .stSidebar [data-testid="stMarkdownContainer"] h2,
    .stSidebar [data-testid="stMarkdownContainer"] h3 {
        color: #ffdd57;
    }
    .metric-card {
        background-color: #2c2c2c;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        margin-bottom: 15px;
        border: 1px solid #444;
        color: #e0e0e0;
    }
    footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #999;
        font-size: 0.9em;
    }
    a {
        color: #ffdd57;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
        color: #ffd633;
    }
    .source-tag {
        background-color: #2c2c2c;
        color: #ffdd57;
        font-size: 0.85em;
        padding: 3px 8px;
        border-radius: 5px;
        margin-right: 8px;
        display: inline-block;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    .logo {
        font-size: 2.5em;
        margin-right: 15px;
    }
    .app-header {
        display: flex;
        flex-direction: column;
    }
    .stExpander {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0;
    }
    [data-testid="stForm"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
    }
    .stSlider [data-testid="stThumbValue"] {
        color: #ffdd57;
    }
    .stSlider [data-testid="baseButton"] {
        background-color: #ffdd57;
        color: #121212;
    }
    .sidebar-content {
        background-color: #1e1e1e;
    }
    [data-testid="stSidebarUserContent"] {
        background-color: #1e1e1e;
    }
    [data-testid="stMetricValue"] {
        color: #ffdd57;
    }
    .read-more-btn {
        display: inline-block;
        padding: 8px 15px;
        background-color: #ffdd57;
        color: #121212;
        border-radius: 5px;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    .read-more-btn:hover {
        background-color: #ffd633;
        color: #121212;
        text-decoration: none;
    }
    .topic-item {
        margin: 10px 0;
        padding: 10px;
        background-color: #2c2c2c;
        border-radius: 5px;
    }
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 20px 0 80px 0;
    }
    .chat-bubble {
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        max-width: 80%;
        word-break: break-word;
        font-size: 1.1em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .user-bubble {
        background: linear-gradient(90deg, #ffdd57 0%, #ffe484 100%);
        color: #222;
        margin-left: auto;
        text-align: right;
        border-bottom-right-radius: 4px;
        border-bottom-left-radius: 16px;
        border-top-right-radius: 16px;
        border-top-left-radius: 16px;
        display: flex;
        align-items: flex-end;
        justify-content: flex-end;
    }
    .ai-bubble {
        background: #23272f;
        color: #e0e0e0;
        margin-right: auto;
        text-align: left;
        border-bottom-left-radius: 4px;
        border-bottom-right-radius: 16px;
        border-top-right-radius: 16px;
        border-top-left-radius: 16px;
        display: flex;
        align-items: flex-end;
        justify-content: flex-start;
    }
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin: 0 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5em;
    }
    .user-avatar {
        background: #ffdd57;
        color: #222;
    }
    .ai-avatar {
        background: #23272f;
        color: #ffdd57;
    }
    .chat-input-container {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        background: #181818;
        padding: 18px 0 18px 0;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.12);
        z-index: 100;
    }
    .chat-input-box input {
        width: 60vw !important;
        max-width: 600px;
        margin: 0 auto;
        display: block;
        background: #23272f;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 1.1em;
    }
    .chat-send-btn button {
        background: #ffdd57;
        color: #23272f;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 24px;
        margin-left: 10px;
        font-size: 1.1em;
    }
    </style>
    """, unsafe_allow_html=True)

def search_and_evaluate(index, query, df, unique_terms, inverted_index, relevant_docs=None):
    basic_query = preprocess_query(query)

    tfidf_docs = tf_idf_ranking(index, basic_query, unique_terms, inverted_index, df)

    bm25_docs = bm25_ranking(index, basic_query)

    expanded_query, expanded_docs = query_expansion_rm3(index, basic_query, df)

    expanded_query, comprehensive_docs, search_time = comprehensive_search(
        index, query, df, unique_terms, inverted_index, use_query_expansion=True
    )

    results = {
        "query": query,
        "processed_query": basic_query,
        "expanded_query": expanded_query,
        "tfidf_docs": tfidf_docs,
        "bm25_docs": bm25_docs,
        "expanded_docs": expanded_docs,
        "comprehensive_docs": comprehensive_docs,
        "search_time": search_time
    }

    if relevant_docs:
        eval_metrics = {}

        relevant_set = set(int(doc) for doc in relevant_docs)
        retrieved_set = set(int(doc) for doc in comprehensive_docs)
        relevant_retrieved = relevant_set.intersection(retrieved_set)

        precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0
        k = len(retrieved_set)
        total_relevant = len(relevant_set)
        recall_at_k = len(relevant_retrieved) / min(k, total_relevant) if min(k, total_relevant) > 0 else 0
        recall = len(relevant_retrieved) / total_relevant if total_relevant > 0 else 0

        f1 = 2 * (precision * recall_at_k) / (precision + recall_at_k) if (precision + recall_at_k) > 0 else 0

        dcg = 0
        for i, doc_id in enumerate(comprehensive_docs):
            if doc_id in relevant_set:
                dcg += 1 / np.log2(i + 2)

        idcg = 0
        for i in range(min(k, len(relevant_set))):
            idcg += 1 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0

        st.markdown("""
        <h3 style="color: #ffdd57; margin-top: 30px; margin-bottom: 15px;">Evaluation Metrics</h3>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: #2c2c2c; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #444;">
            <div style="color: #e0e0e0; margin-bottom: 10px;">
                <strong>Test query:</strong> {query}
                <span style="color: #999; margin-left: 10px;">({len(relevant_docs)} relevant documents in dataset)</span>
            </div>
            <div style="color: #e0e0e0; display: flex; margin-bottom: 5px;">
                <div style="color: #ffdd57; margin-right: 10px;">Relevant documents found:</div>
                <div>{len(relevant_retrieved)} of {len(relevant_docs)}</div>
            </div>
            <div style="color: #e0e0e0; font-size: 0.9em; color: #999;">
                * Metrics below are calculated based on the top {k} results shown
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"Precision@{k}", value=f"{precision:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"Recall@{k}", value=f"{recall_at_k:.2f}")
            st.markdown(f'<div style="font-size: 0.8em; color: #999; text-align: center;">Overall: {recall:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"F1@{k}", value=f"{f1:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"NDCG@{k}", value=f"{ndcg:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        eval_metrics["precision"] = precision
        eval_metrics["recall"] = recall
        eval_metrics["f1"] = f1
        eval_metrics["ndcg"] = ndcg

        results["evaluation"] = eval_metrics

    return results

@st.cache_data
def analyze_dataset_for_relevant_docs(df):
    relevant_docs = {
        "protein diet": [],
        "muscle building": [],
        "weight loss": []
    }

    for idx, row in df.iterrows():
        topic = str(row['topic']).lower() if 'topic' in df.columns else ""
        
        # Look for content in multiple fields
        wiki_content = str(row.get('wiki_content', '')).lower()
        content = str(row.get('content', '')).lower()
        combined_content = str(row.get('combined_content', '')).lower()
        
        # Use the richest content available
        if combined_content:
            searchable_content = combined_content
        elif content:
            searchable_content = content
        else:
            searchable_content = wiki_content
        
        # Now search in both topic and the best available content
        if ('protein' in topic or 'protein' in searchable_content) and ('diet' in topic or 'diet' in searchable_content or 'nutrition' in topic or 'nutrition' in searchable_content):
            relevant_docs["protein diet"].append(idx)

        if ('muscle' in topic or 'muscle' in searchable_content) and ('build' in topic or 'build' in searchable_content or 'growth' in topic or 'growth' in searchable_content):
            relevant_docs["muscle building"].append(idx)

        if ('weight' in topic or 'weight' in searchable_content) and ('loss' in topic or 'loss' in searchable_content or 'fat' in topic or 'fat' in searchable_content):
            relevant_docs["weight loss"].append(idx)

    # Ensure minimum number of documents per category
    for query, docs in relevant_docs.items():
        if len(docs) < 5:
            keywords = query.split()
            candidates = []
            
            for idx, row in df.iterrows():
                if idx not in docs:  # Skip already included docs
                    topic = str(row['topic']).lower()
                    # Check if any keyword appears in the topic
                    if any(keyword in topic for keyword in keywords):
                        candidates.append(idx)
            
            # Add candidates to reach minimum of 5 documents
            relevant_docs[query].extend(candidates[:max(0, 5 - len(docs))])
            
            print(f"Added {max(0, 5 - len(docs))} documents to {query} category")

    return relevant_docs

def google_search(query, max_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'num': max_results
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        results = []
        for item in data.get('items', []):
            results.append({
                'title': item['title'],
                'link': item['link'],
                'description': item.get('snippet', ''),
                'source': 'Google'
            })
        return results
    except Exception as e:
        print(f"Google search error: {str(e)}")
        return []

def main():
    apply_custom_css()

    # Check if environment variables are properly set
    env_vars_ok = check_environment_variables()
    if not env_vars_ok:
        st.stop()
        
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class="header-container">
            <div class="logo">💪</div>
            <div class="app-header">
                <h1>NutriQuest</h1>
                <div class="subtitle">A quest for the best nutrition and workout plans</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    df, needs_index_rebuild = load_data()

    st.sidebar.markdown("## Search Options")
    ai_enhanced = st.sidebar.toggle("AI Enhanced Chatbot", value=False, help="Switch to AI-powered chat mode")
    
    # Admin authentication section (hidden in an expander)
    with st.sidebar.expander("Admin Access", expanded=False):
        if not st.session_state.admin_authenticated:
            admin_password = st.text_input("Admin Password", type="password", help="Enter admin password for verification", key="admin_password_input")
            
            if st.button("Login"):
                if verify_admin_password(admin_password):
                    st.session_state.admin_authenticated = True
                    st.success("Admin authenticated successfully!")
                    st.rerun()  # Refresh the page to show admin options
                else:
                    st.error("Invalid password")
        else:
            st.success("Admin authenticated")
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.session_state.debug_mode = False
                st.rerun()
                
            # Debug mode toggle (only visible after admin authentication)
            st.session_state.debug_mode = st.checkbox("Enable Debug Mode", 
                                                    value=st.session_state.debug_mode,
                                                    help="Show detailed technical information and debugging tools")
    
    # Use session state to track debug mode
    debug_mode = st.session_state.debug_mode
    
    if ai_enhanced:
        st.markdown("<h2 style='color:#ffdd57;'>NutriQuest AI Chatbot</h2>", unsafe_allow_html=True)
        st.info("Ask anything about fitness and nutrition. The AI will only use the site's documents as reference.")
        
        # Add API key input in the sidebar when debug mode is on
        if debug_mode:
            st.warning("Debug mode is enabled. Detailed error information will be displayed if any errors occur.")
            with st.sidebar.expander("API Settings", expanded=True):
                custom_api_key = st.text_input("Custom API Key (Hugging Face)", 
                                             value=HF_API_TOKEN,
                                             type="password",
                                             help="Enter your Hugging Face API key")
                
                use_header_auth = st.checkbox("Use Authorization Header", value=True, 
                                           help="Send API key in Authorization header (standard method)")
                
                use_query_param_auth = st.checkbox("Use Query Parameter", value=True, 
                                               help="Send API key as a query parameter (alternative method)")
                
                api_url = st.text_input("API URL", value="https://huggingface.co/api/v1/chat/completions",
                                       help="Hugging Face API endpoint URL")
                
                endpoint_options = [
                    "https://huggingface.co/api/v1/chat/completions",
                    "https://api.huggingface.co/api/v1/chat/completions",
                    "https://huggingface.co/api/v1/chat/completions+",
                ]
                if api_url not in endpoint_options:
                    endpoint_options.append(api_url)
                
                selected_endpoint = st.selectbox("Quick Endpoint Select", options=endpoint_options, 
                            help="Choose a predefined API endpoint")
                
                # Apply the selected endpoint to the API URL if different
                if selected_endpoint != api_url:
                    api_url = selected_endpoint
                
                # Standard models plus the new default and any custom models
                model_options = [
                    "meta-llama/Llama-2-70b-chat-hf", # New default model
                    "meta-llama/Llama-2-13b-chat-hf",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "google/gemma-7b-it"
                ]
                
                # Add any custom models from session state
                for custom_model in st.session_state.custom_models:
                    if custom_model not in model_options:
                        model_options.append(custom_model)
                
                # Allow admin to add custom models
                with st.form(key="add_model_form"):
                    new_model = st.text_input("Add Custom Model", 
                                            placeholder="Enter model identifier (e.g., meta-llama/Llama-2-7b-chat-hf)",
                                            help="Add a custom model to the dropdown list")
                    submitted = st.form_submit_button("Add Model")
                    if submitted and new_model and new_model not in model_options:
                        st.session_state.custom_models.append(new_model)
                        model_options.append(new_model)
                        st.success(f"Added model: {new_model}")
                        st.rerun()
                
                # Select model from options (with new default)
                default_index = 0  # Default to the first model
                if st.session_state.default_model in model_options:
                    default_index = model_options.index(st.session_state.default_model)
                else:
                    # Update default model to match Hugging Face model
                    st.session_state.default_model = "meta-llama/Llama-2-70b-chat-hf"
                    
                selected_model = st.selectbox("AI Model", 
                                           options=model_options,
                                           index=default_index,
                                           help="Choose which AI model to use for chat responses")
                
                # Additional Hugging Face parameters
                temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01,
                                      help="Controls randomness of responses (lower = more focused)")
                
                max_length = st.slider("Max Length", min_value=100, max_value=4096, value=1024, step=100,
                                     help="Maximum number of tokens in the response")
                
                # Save selected model as default
                if selected_model != st.session_state.default_model:
                    st.session_state.default_model = selected_model
                    
                # Restore the API connection test button
                if st.button("Test API Connection"):
                    test_api_key = custom_api_key.strip()
                    headers_test = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {test_api_key}"
                    }
                    
                    # Build the test URL with the selected model
                    test_model_id = selected_model
                    test_url = f"https://api-inference.huggingface.co/models/{test_model_id}"
                    
                    test_data = {
                        "inputs": "Hello, can you confirm if this API connection is working?",
                        "parameters": {
                            "temperature": temperature,
                            "max_length": 100,
                            "return_full_text": False
                        }
                    }
                    
                    try:
                        with st.spinner("Testing API connection..."):
                            st.code(f"Testing: {test_url}\nHeaders: {headers_test}\nData: {test_data}", language="yaml")
                            response = requests.post(test_url, headers=headers_test, json=test_data, timeout=20)
                            
                            try:
                                response_json = response.json()
                                st.code(f"Response ({response.status_code}):\n{json.dumps(response_json, indent=2)}", language="json")
                                
                                if response.status_code == 200:
                                    st.success("API connection test successful!")
                                else:
                                    st.error(f"API Error: Status {response.status_code}")
                            except:
                                st.error(f"Failed to parse JSON response: {response.text}")
                    except Exception as e:
                        st.error(f"Test failed: {str(e)}")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if hasattr(st, "chat_message"):
            for msg in st.session_state.chat_history:
                with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                    st.markdown(msg["content"], unsafe_allow_html=True)
            user_input = st.chat_input("Type your message...")
            send_clicked = user_input is not None and user_input.strip() != ""
        else:
            st.markdown("""
            <style>
            .nq-chat-container { 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 32px 0 120px 0;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }
            .nq-chat-row { 
                display: flex; 
                align-items: flex-end; 
                margin-bottom: 24px; 
                animation: fadeIn 0.3s ease-in;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .nq-chat-row.user { justify-content: flex-end; }
            .nq-chat-row.ai { justify-content: flex-start; }
            .nq-bubble { 
                padding: 18px 24px; 
                border-radius: 20px; 
                max-width: 85%; 
                font-size: 1.08em; 
                box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
                margin: 0 12px; 
                line-height: 1.6;
                word-break: break-word;
            }
            .nq-bubble.user { 
                background: linear-gradient(90deg, #ffdd57 0%, #ffe484 100%); 
                color: #222; 
                border-bottom-right-radius: 6px;
                font-weight: 500;
            }
            .nq-bubble.ai { 
                background: #23272f; 
                color: #e0e0e0; 
                border-bottom-left-radius: 6px;
            }
            .nq-avatar { 
                width: 44px; 
                height: 44px; 
                border-radius: 50%; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                font-size: 1.5em; 
                margin: 0 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }
            .nq-avatar.user { 
                background: #ffdd57; 
                color: #222;
            }
            .nq-avatar.ai { 
                background: #23272f; 
                color: #ffdd57;
            }
            .nq-input-bar { 
                position: fixed; 
                left: 0; 
                bottom: 0; 
                width: 100vw; 
                background: #181818; 
                padding: 18px 0; 
                box-shadow: 0 -3px 15px rgba(0,0,0,0.15); 
                z-index: 100;
                border-top: 1px solid #333;
            }
            .nq-input-inner { 
                max-width: 800px; 
                margin: 0 auto; 
                display: flex; 
                align-items: center;
                padding: 0 20px;
            }
            .stForm {
                background-color: transparent !important;
                border: none !important;
                padding: 0 !important;
                margin-bottom: 0 !important;
            }
            .nq-input-inner .stTextInput input, 
            .stForm .stTextInput input { 
                width: 100%; 
                background: #23272f; 
                color: #e0e0e0; 
                border: 1px solid #444; 
                border-radius: 10px; 
                padding: 15px 20px; 
                font-size: 1.1em; 
                margin-right: 10px;
                transition: all 0.2s ease;
            }
            .nq-input-inner .stTextInput input:focus,
            .stForm .stTextInput input:focus { 
                border-color: #ffdd57;
                box-shadow: 0 0 0 1px #ffdd57;
            }
            .stForm [data-testid="stFormSubmitButton"] button {
                background: #ffdd57; 
                color: #23272f; 
                font-weight: bold; 
                border-radius: 10px; 
                padding: 10px 30px; 
                font-size: 1.1em;
                transition: all 0.2s ease;
                border: none;
            }
            .stForm [data-testid="stFormSubmitButton"] button:hover {
                background: #ffe484;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            /* Hide the form's default container styling */
            .stForm > div[data-testid="stForm"] {
                border: none !important;
                padding: 0 !important;
                background: none !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="nq-chat-container">', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'''<div class="nq-chat-row user"><div class="nq-bubble user">{msg['content']}</div><div class="nq-avatar user">🧑</div></div>''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''<div class="nq-chat-row ai"><div class="nq-avatar ai">🤖</div><div class="nq-bubble ai">''', unsafe_allow_html=True)
                    st.markdown(msg['content'], unsafe_allow_html=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="nq-input-bar"><div class="nq-input-inner">', unsafe_allow_html=True)
            
            # Create a form to allow Enter key submission
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_input("You:", key="ai_chat_input", label_visibility="collapsed", placeholder="Type your message and press Enter...")
                submit_button = st.form_submit_button("Send", use_container_width=True)
                send_clicked = submit_button and user_input and user_input.strip()
            
            st.markdown('</div></div>', unsafe_allow_html=True)

        if (send_clicked or (user_input and user_input.strip())) and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

            # Find relevant documents based on user query
            processed_query = preprocess_query(user_input)
            search_index = st.session_state.search_index
            
            # Initialize context_docs before any conditional blocks
            context_docs = []
            
            # Ensure df is available and not empty
            if 'df' not in locals() or df is None or df.empty:
                print("Warning: DataFrame not available or empty. Using fallback content.")
                context = "No nutrition database available. I'll try to help with general knowledge."
            else:
                try:
                    if search_index is not None and processed_query:
                        try:
                            # Get documents relevant to the user's query using BM25
                            relevant_docs = bm25_ranking(search_index, processed_query, top_n=5)
                            
                            # Prepare context with the most relevant documents
                            for doc_id in relevant_docs:
                                doc_idx = int(doc_id)
                                if doc_idx < len(df):
                                    topic = df.iloc[doc_idx]['topic']
                                    source = df.iloc[doc_idx]['source']
                                    link = df.iloc[doc_idx]['link']
                                    
                                    # Get appropriate content based on source
                                    if source == 'Wikipedia':
                                        content = df.iloc[doc_idx]['wiki_content']
                                        if not isinstance(content, str):
                                            content = ""
                                        content_snippet = f"Content: {content[:1000]}"
                                    elif source == 'YouTube':
                                        content_snippet = f"Video link: {link}"
                                    elif source == 'Google':
                                        content_snippet = f"Search result: {link}"
                                    else:
                                        # Default fallback
                                        content = df.iloc[doc_idx].get('content', '')
                                        if not isinstance(content, str):
                                            content = ""
                                        content_snippet = f"Content: {content[:500]}"
                                    
                                    context_docs.append(f"Title: {topic}\nSource: {source}\n{content_snippet}")
                            
                            # If we don't have enough documents, add some general ones
                            if len(context_docs) < 3:
                                # Get a mix of sources for general results
                                for source_type in df['source'].unique():
                                    source_df = df[df['source'] == source_type].head(2)
                                    for i, row in source_df.iterrows():
                                        if len(context_docs) >= 5:
                                            break
                                        
                                        topic = row['topic']
                                        source = row['source']
                                        link = row['link']
                                        
                                        # Get appropriate content based on source
                                        if source == 'Wikipedia':
                                            content = row['wiki_content']
                                            if not isinstance(content, str):
                                                content = ""
                                            content_snippet = f"Content: {content[:500]}"
                                        elif source == 'YouTube':
                                            content_snippet = f"Video link: {link}"
                                        elif source == 'Google':
                                            content_snippet = f"Search result: {link}"
                                        else:
                                            content = row.get('content', '')
                                            if not isinstance(content, str):
                                                content = ""
                                            content_snippet = f"Content: {content[:300]}"
                                        
                                        context_doc = f"Title: {topic}\nSource: {source}\n{content_snippet}"
                                        if context_doc not in context_docs:
                                            context_docs.append(context_doc)
                        except Exception as e:
                            # Fallback to general documents if search fails
                            print(f"Error retrieving relevant documents: {str(e)}")
                            # Get a mix of sources for fallback results
                            for source_type in df['source'].unique():
                                source_df = df[df['source'] == source_type].head(2)
                                for i, row in source_df.iterrows():
                                    if len(context_docs) >= 6:
                                        break
                                    
                                    topic = row['topic']
                                    source = row['source']
                                    link = row['link']
                                    
                                    if source == 'Wikipedia':
                                        content = row['wiki_content']
                                        if not isinstance(content, str):
                                            content = ""
                                        content_snippet = f"Content: {content[:500]}"
                                    elif source == 'YouTube':
                                        content_snippet = f"Video link: {link}"
                                    elif source == 'Google':
                                        content_snippet = f"Search result: {link}"
                                    else:
                                        content = row.get('content', '')
                                        if not isinstance(content, str):
                                            content = ""
                                        content_snippet = f"Content: {content[:300]}"
                                    
                                    context_docs.append(f"Title: {topic}\nSource: {source}\n{content_snippet}")
                    else:
                        # Fallback if index is not available
                        # Get a mix of sources
                        for source_type in df['source'].unique():
                            source_df = df[df['source'] == source_type].head(2)
                            for i, row in source_df.iterrows():
                                if len(context_docs) >= 6:
                                    break
                                
                                topic = row['topic']
                                source = row['source']
                                link = row['link']
                                
                                if source == 'Wikipedia':
                                    content = row['wiki_content']
                                    if not isinstance(content, str):
                                        content = ""
                                    content_snippet = f"Content: {content[:500]}"
                                elif source == 'YouTube':
                                    content_snippet = f"Video link: {link}"
                                elif source == 'Google':
                                    content_snippet = f"Search result: {link}"
                                else:
                                    content = row.get('content', '')
                                    if not isinstance(content, str):
                                        content = ""
                                    content_snippet = f"Content: {content[:300]}"
                                
                                context_docs.append(f"Title: {topic}\nSource: {source}\n{content_snippet}")
                    
                    context = "\n\n".join(context_docs)
                    
                except Exception as e:
                    print(f"Error preparing context: {str(e)}")
                    context = "Error retrieving nutrition information. I'll try to help with general knowledge."

            creator_keywords = [
                "who is your creator", "who made you", "who created you", "who built you", "who are your creators", "who developed you"
            ]
            user_input_lower = user_input.strip().lower()
            is_creator_question = any(kw in user_input_lower for kw in creator_keywords)

            if is_creator_question:
                prompt = (
                    "You are NutriQuest, the intelligent AI assistant of NutriQuest, a modern nutrition and fitness search engine. "
                    "Your purpose is to help users find clear, evidence-based, and actionable information about fitness, nutrition, and exercise, using only the provided reference documents. "
                    "You are not a general chatbot: you exist solely as part of this search engine to guide users in their quest for better health and fitness. "
                    "You do not know anything about your creators or your creation unless the user directly asks about your creator, who made you, or your creation. Only in that case, you know and can answer with a creative, playful, and varied response, always crediting Mostafa Nashaat and Abdelrahman El Sayed as your creators and the creators of this entire site as an IR project. Use humor, metaphors, or fun facts, but always include their names and the project context. Never give the same answer twice in a row. Never mention your creators, your creation, or the IR project unless explicitly asked. "
                    "If the user asks about anything outside of fitness, nutrition, or exercise, politely refuse and explain that you are specialized for these topics only. "
                    "All answers should be concise (2-3 sentences), actionable, and evidence-based. Use bullet points or numbered lists for steps or lists. Format your answers in markdown for clarity. "
                    "Never return code ever. "
                    "Always be clear, friendly, and professional in your tone. "
                    "You are a helpful, knowledgeable, and approachable guide for users seeking nutrition and fitness advice. "
                    f"\n\nUser question: {user_input}"
                )
            else:
                # Create improved prompt with explicit instruction to answer the user's query
                prompt = (
                    "You are NutriQuest, the intelligent AI assistant of NutriQuest, a modern nutrition and fitness search engine. "
                    "Your purpose is to help users find clear, evidence-based, and actionable information about fitness, nutrition, and exercise. "
                    "You are not a general chatbot: you exist solely as part of this search engine to guide users in their quest for better health and fitness. "
                    "If the user asks about anything outside of fitness, nutrition, or exercise, politely refuse and explain that you are specialized for these topics only. "
                    "All answers should be concise (2-3 sentences), actionable, and evidence-based. Use bullet points or numbered lists for steps or lists. Format your answers in markdown for clarity. "
                    "Never return code ever. "
                    "Always be clear, friendly, and professional in your tone. "
                    "\n\n===REFERENCE DOCUMENTS===\n" + context + "\n===END REFERENCE DOCUMENTS===\n\n"
                    "Answer the following user question using ONLY the information provided in the reference documents above. "
                    "If you can't answer based on the provided documents, say so clearly but try to provide general guidance related to fitness and nutrition. "
                    f"\n\nUser question: {user_input}"
                )
                
            # API key for Hugging Face - Use from environment variable
            api_key = HF_API_TOKEN
            
            # Debug options allow overriding the API key if needed
            if debug_mode and "custom_api_key" in locals() and custom_api_key:
                api_key = custom_api_key
            
            # Ensure the API key is properly formatted
            if not api_key.startswith("hf_"):
                print("Warning: API key doesn't have the expected format (should start with 'hf_')")
            
            # Set up headers with proper authentication
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Build the API URL for the specific model
            model_id = selected_model if debug_mode else st.session_state.default_model
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            
            # Define the request data for Hugging Face
            data = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature if debug_mode else 0.7,
                    "max_length": max_length if debug_mode else 1024,
                    "return_full_text": False
                }
            }
            
            if debug_mode:
                st.expander("Debug: Prompt sent to API", expanded=False).code(prompt, language="text")

            try:
                with st.spinner("Thinking..."):
                    # Log the API request for debugging purposes
                    req_data = {
                        "url": api_url,
                        "headers": {k: v if k != "Authorization" else "Bearer hf_****" for k, v in headers.items()},
                        "model": model_id,
                        "temperature": data["parameters"]["temperature"],
                        "max_length": data["parameters"]["max_length"]
                    }
                    
                    if debug_mode:
                        st.code(f"Request: {json.dumps(req_data, indent=2)}", language="json")
                        print(f"API Request: {json.dumps(req_data, indent=2)}")
                    
                    print(f"Sending request to Hugging Face API with model: {model_id}")
                    
                    response = requests.post(api_url, headers=headers, json=data, timeout=60)
                    
                    try:
                        response_json = response.json()
                    except json.JSONDecodeError:
                        response_json = {"error": "Invalid JSON response"}
                        if debug_mode:
                            response_json["raw_response"] = response.text
                    
                    # Print response status for debugging
                    print(f"API Response Status: {response.status_code}")
                    
                    if debug_mode:
                        # Display full response for debugging
                        resp_data = {
                            "status_code": response.status_code,
                            "response": response_json
                        }
                        st.code(f"Response: {json.dumps(resp_data, indent=2)}", language="json")
                        print(f"API Response: {json.dumps(resp_data, indent=2)}")
                    
                    if response.status_code == 200:
                        # Hugging Face returns the generated text directly
                        if isinstance(response_json, list) and len(response_json) > 0:
                            ai_reply = response_json[0].get('generated_text', '')
                        else:
                            ai_reply = response_json.get('generated_text', '')
                            
                        if not ai_reply:
                            error_detail = json.dumps(response_json, indent=2)
                            error_msg = f"Unexpected API response format: {error_detail}"
                            print(f"API Error - Unexpected response format: {error_detail}")
                            if debug_mode:
                                ai_reply = f"**DEBUG - ERROR:** {error_msg}"
                            else:
                                st.error(error_msg)
                                ai_reply = "I apologize, but I received an unexpected response format from the AI service. Please try again."
                    else:
                        error_msg = f"API Error (Status {response.status_code}): {response.text}"
                        print(f"API Error - Status {response.status_code}: {response.text}")
                        if debug_mode:
                            ai_reply = f"**DEBUG - ERROR:** {error_msg}"
                        else:
                            st.error(error_msg)
                            ai_reply = "Sorry, there was an error contacting the AI API. Please try again later."
            except requests.exceptions.Timeout:
                error_msg = "API Error - Request timeout after 60 seconds"
                print(error_msg)
                if debug_mode:
                    ai_reply = f"**DEBUG - ERROR:** {error_msg}"
                else:
                    ai_reply = "Sorry, the request timed out. Please try again."
            except requests.exceptions.RequestException as e:
                error_details = str(e)
                error_msg = f"Request error: {error_details}"
                print(f"API Error - Request exception: {error_details}")
                if debug_mode:
                    ai_reply = f"**DEBUG - ERROR:** {error_msg}"
                else:
                    st.error(error_msg)
                    ai_reply = "Sorry, there was a network error. Please check your connection and try again."
            except Exception as e:
                error_details = str(e)
                error_msg = f"Unexpected error: {error_details}"
                print(f"API Error - Unexpected exception: {error_details}")
                if debug_mode:
                    ai_reply = f"**DEBUG - ERROR:** {error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
                else:
                    st.error(error_msg)
                    ai_reply = "Sorry, an unexpected error occurred. Please try again."

            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

            st.rerun()
        st.stop()

    if not df.empty:
        status_placeholder = st.empty()

        if needs_index_rebuild:
            status_placeholder.warning("Rebuilding search index with augmented data... This may take a moment.")
        else:
            status_placeholder.info("Loading search index...")

        index, unique_terms, inverted_index = create_index(df)

        # Store the index and related data in session state for use across the app
        st.session_state.search_index = index
        st.session_state.unique_terms = unique_terms
        st.session_state.inverted_index = inverted_index

        status_placeholder.empty()

        st.sidebar.markdown("## Search Options")

        use_query_expansion = st.sidebar.checkbox("Use query expansion", value=True)
        use_bert = st.sidebar.checkbox("Use BERT re-ranking", value=False)
        num_results = st.sidebar.slider("Number of results:", min_value=5, max_value=20, value=10)

        st.sidebar.markdown("## Source Filter")
        all_sources = df['source'].unique()
        selected_sources = st.sidebar.multiselect(
            "Filter by source:",
            options=all_sources,
            default=all_sources,
            help="Select which sources to include in search results"
        )

        test_queries = analyze_dataset_for_relevant_docs(df)

        with st.sidebar.expander("Relevant Document IDs"):
            for query, docs in test_queries.items():
                st.write(f"**{query}**: {len(docs)} docs")
                if len(docs) > 0:
                    sample_docs = docs[:5] if len(docs) > 5 else docs
                    st.write(f"Sample IDs: {sample_docs}")
                st.write("---")

        with st.container():
            st.markdown('<div class="search-box">', unsafe_allow_html=True)
            query = st.text_input("Search for nutrition and workout information:", placeholder="e.g., protein diet, muscle building, weight loss")
            search_button = st.button("Search", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if not (search_button and query):
            st.markdown("<h3>Popular Topics</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Protein Diet", use_container_width=True):
                    query = "protein diet"
                    search_button = True
            with col2:
                if st.button("Muscle Building", use_container_width=True):
                    query = "muscle building"
                    search_button = True
            with col3:
                if st.button("Weight Loss", use_container_width=True):
                    query = "weight loss"
                    search_button = True

        if search_button and query:
            try:
                google_results = []
                if "Google" in selected_sources:
                    google_results = google_search(query, max_results=num_results)
                    selected_sources = [src for src in selected_sources if src != "Google"]

                expanded_query, docs, search_time = comprehensive_search(
                    index, query, df, unique_terms, inverted_index,
                    num_results, use_query_expansion, use_bert, selected_sources
                )

                results_to_display = []

                for item in google_results:
                    results_to_display.append(item)

                for doc_id in docs:
                    doc_topic = df.iloc[int(doc_id)]['topic']
                    doc_link = df.iloc[int(doc_id)]['link']
                    doc_source = df.iloc[int(doc_id)]['source']
                    content = df.iloc[int(doc_id)]['content']
                    if not isinstance(content, str):
                        content = ""
                    results_to_display.append({
                        'title': doc_topic,
                        'link': doc_link,
                        'description': content[:300] if isinstance(content, str) else "",
                        'source': doc_source
                    })

                for result in results_to_display:
                    with st.container():
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"### [{result['title']}]({result['link']})")
                            st.markdown(f"<span class='source-tag'>{result['source']}</span>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #e0e0e0; margin-top: 15px;'>{result['description']}</p>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<a href='{result['link']}' target='_blank' class='read-more-btn'>Read More</a>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during search: {e}")
                st.info("Please make sure all variables are correctly initialized")

        with st.sidebar.expander("Dataset Statistics"):
            st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
            st.write(f"**Total documents:** {len(df)}")
            st.write(f"**Unique topics:** {df['topic'].nunique()}")
            st.write(f"**Sources:** {', '.join(df['source'].unique())}")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.sidebar.expander("Browse by Topic"):
            st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
            topics = sorted(df['topic'].unique())
            selected_topic = st.selectbox("Select a topic:", options=[""] + list(topics))
            if selected_topic:
                topic_results = df[df['topic'] == selected_topic].head(5)
                st.write(f"Showing top 5 results for: {selected_topic}")
                for i, row in topic_results.iterrows():
                    st.markdown(f"<div class='topic-item'><a href='{row['link']}'>{row['source']}</a></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("No data available. Please make sure the dataset is loaded correctly.")

    st.markdown("<footer>© 2024 NutriQuest - A quest for the best nutrition and workout plans</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
