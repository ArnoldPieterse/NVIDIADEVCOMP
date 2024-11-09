import os
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import faiss.swigfaiss as faiss
from googlesearch import search
import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import base64, json, shutil
from pathlib import Path
import pdfkit
import numpy as np
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Set the NVIDIA API key
os.environ["NVIDIA_API_KEY"] = "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"

# Configure settings for the application
Settings.text_splitter = SentenceSplitter(chunk_size=500)
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.llm = NVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct")

# Check if NVIDIA API key is set as an environment variable
if os.getenv("NVIDIA_API_KEY") is None:
    raise ValueError("Please set the NVIDIA_API_KEY environment variable")

# Initialize global variables for the index and query engine
index = None
query_engine = None
documents_bool = False
core_documents = []

def setup_chrome_driver():
    """Initialize Chrome driver with appropriate options"""
    try:
        chromedriver = shutil.which("chromedriver")
        if not chromedriver:
            raise RuntimeError("chromedriver not found on PATH")
        
        options = Options()
        # Basic options
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        
        # SSL/Security options
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--allow-insecure-localhost")
        options.add_argument("--ignore-ssl-errors")
        options.add_argument("--disable-web-security")
        
        # Performance options
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.page_load_strategy = 'eager'
        
        # Initialize driver with extended timeout
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(60)
        driver.implicitly_wait(20)
        
        return driver
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chrome driver: {str(e)}")

def send_devtools(driver, cmd, params={}):
    """Send DevTools command to Chrome using CDP"""
    try:
        # Get the DevTools endpoint from driver
        debugger_url = driver.capabilities.get('goog:chromeOptions', {}).get('debuggerAddress')
        if not debugger_url:
            debugger_url = "127.0.0.1:9222"  # Default Chrome debugging port
            
        # Construct the DevTools URL
        devtools_url = f"http://{debugger_url}/json"
        
        # Send command using driver's CDP
        result = driver.execute_cdp_cmd(cmd, params)
        
        if not result:
            raise Exception("Empty response from DevTools")
        return result
        
    except Exception as e:
        raise RuntimeError(f"DevTools command failed: {str(e)}")

def url_to_pdf(url, output_path, driver=None):
    """Convert URL to PDF using pdfkit"""
    try:
        # Configure pdfkit options
        options = {
            'quiet': '',
            'no-images': '',
            'disable-javascript': '',
            'encoding': 'UTF-8',
            'print-media-type': '',
            'no-outline': '',
            'page-size': 'A4'
        }
        
        # Convert URL to PDF
        pdfkit.from_url(url, output_path, options=options)
        return True
        
    except Exception as e:
        print(f"Error converting {url}: {str(e)}")
        return False

# Function to get file names from file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load documents and create the index
def load_documents(file_objs):
    global index, query_engine
    try:
        if not file_objs:
            return "No files selected"
        
        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"No documents found"
        
        vector_store = VectorStoreIndex.from_documents(documents)
        query_engine = vector_store.as_query_engine()
        return f"Loaded {len(documents)} documents"
    except Exception as e:
        return f"Error: {str(e)}"

def add_documents(documents, limit):
    global query_engine, index
    try:
        if not documents:
            return "No documents provided"
        
        if query_engine is None:
            # Create new index and query engine from documents
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
        else:
            # Get the underlying index from query engine
            if hasattr(query_engine, 'retriever'):
                # Access the index through service context
                if hasattr(query_engine.retriever, '_index'):
                    index = query_engine.retriever._index
                elif hasattr(query_engine.retriever, '_vector_store'):
                    # Try accessing through vector store
                    index = query_engine.retriever._vector_store.index
                else:
                    # Create new index if can't access existing
                    print("Creating new index from documents")
                    all_docs = []
                    if index:
                        all_docs.extend(index.docstore.docs.values())
                    all_docs.extend(documents)
                    # all_docs can only house the latest 5 documents, remove the rest to keep the context small
                    all_docs = all_docs[-limit:]
                    index = VectorStoreIndex.from_documents(all_docs)
                    
                # Update nodes
                index.insert_nodes(documents)
                query_engine = index.as_query_engine()
            else:
                # Create new index with combined documents
                all_docs = []
                if index:
                    all_docs.extend(index.docstore.docs.values())
                all_docs.extend(documents)
                index = VectorStoreIndex.from_documents(all_docs)
                query_engine = index.as_query_engine()
            
        return f"Added {len(documents)} documents"
    except Exception as e:
        print(f"Debug - Query engine type: {type(query_engine)}")
        if hasattr(query_engine, 'retriever'):
            print(f"Debug - Retriever type: {type(query_engine.retriever)}")
        print(f"Debug - Document type: {type(documents[0])}")
        return f"Error: {str(e)}"

# Wrapper function to handle file uploads
def handle_file_upload(file_input):
    return load_documents(file_objs=file_input)
    """return load_documents(file_objs=file_input, urls=None)"""

# Function to handle chat interactions
def chat(message, history):
    global query_engine
    if query_engine is None:
        return history + [("No documents loaded, searching web", None)]
    
    try:
        # Search for relevant links
        """links = perform_google_search(message)"""
        
        # Load documents if needed
        """if query_engine is None:
            load_result = load_documents(urls=links)
            if "Error" in load_result:
                return history + [(message, f"Error loading content: {load_result}")]"""
        
            # Process query
        response = query_engine.query(message)
        return history + [(message, response)]   
    except Exception as e:
        return history + [(message, f"Error processing query: {str(e)}")]

def perform_google_search(query, num_results=5):
    try:
        links = []
        for j in search(query, tld="co.in", num=num_results, stop=num_results, pause=2):
            links.append(j)
        return links
    except Exception as e:
        return []

def load_search_results(query):
    """Load documents from Google search results"""
    try:
        # Perform search
        print(f"Searching for: {query}")
        search_results = perform_google_search(query)
        print(f"Found {len(search_results)} URLs")
        
        if not search_results:
            return "No search results found", [], []
        
        # If search results are more than 3, limit to top 3
        if len(search_results) > 2:
            search_results = search_results[:3]

        # Create temporary directory and load documents
        temp_dir = tempfile.mkdtemp()
        documents = []
        
        for i, url in enumerate(search_results):
            print(f"Converting {url} to PDF...")
            pdf_path = os.path.join(temp_dir, f"webpage_{i}.pdf")
            if url_to_pdf(url, pdf_path):
                print(f"Loading PDF {pdf_path}")
                documents.extend(SimpleDirectoryReader(input_files=[pdf_path]).load_data())
                
        print(f"Loaded {len(documents)} documents")
        return "Documents loaded from search", search_results, documents
    except Exception as e:
        return f"Error: {str(e)}", [], []

def process_external_queries(query_engine, response):
    """
    Process external queries concurrently and return summarized findings.
    
    Args:
        query_engine: The query engine instance
        response: The initial response to generate queries from
    
    Returns:
        str: Summarized external findings
    """
    try:
        # Get external queries
        external = query_engine.query(
            f"Create a query that could question or request more information about "
            f"{{{str(response)}}} while being absolutely gratefull. Thanking. "
            f"Place each search query in <> and separate them with comma. Give short answer."
        )
        
        # Extract queries between < >
        external_temp = []
        queries = [
            q[q.find("<")+1:q.find(">")]
            for q in str(external).split("\n")
            if "<" in q and ">" in q
        ]
        
        # Process queries concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Map queries to futures
            future_to_query = {
                executor.submit(load_search_results, query): query 
                for query in queries
            }
            
            # Process results as they complete
            for future in future_to_query:
                status, search_results, docs = future.result()
                if docs:
                    load_result = add_documents(docs, 2)
                    if "Error" not in load_result:
                        query_response = query_engine.query(
                            f"Check and correct this and only provide desired answer "
                            f"as short as possible for query: {{{future_to_query[future]}}}."
                        )
                        external_temp.append(str(query_response))

        # Summarize and return findings
        if external_temp:
            return query_engine.query(f"Summarize {{{' '.join(external_temp)}}} in a short answer.")
        return "No external information found"

    except Exception as e:
        print(f"Error processing external queries: {str(e)}")
        return f"Error: {str(e)}"

def stream_responses(message, history):
    global query_engine
    
    try:
        # Initial search and document loading
        status, search_results, documents = load_search_results(message)
        if documents:
            core_documents.extend(documents)
            load_result = add_documents(documents, 5)
            if "Error" in load_result:
                yield history + [(message, f"Error loading content: {load_result}")]
                return

        # Batch initial queries
        queries = {
            'main': f"History: {str(history)}. New query: {message}" if history else message,
            'context': f"Give me a descriptive answer to {{{message}}}.",
        }
        
        responses = {}
        for key, query in queries.items():
            responses[key] = query_engine.query(query)
        
        # Process main response
        response = query_engine.query(f"Check and correct this and only provide desired answer as short as possible for query: {{{message}}}. That got response: {{{str(responses['main'])}}}.")

        # Batch emotional responses
        emotion_queries = {
            'animal': f"You are a random animal, what animal, how does {{{str(response)}}} make you feel, or affect you? Give short answer.",
            'heart': f"Make {{{str(response)}}} absolute positive, point out the best in it and be [happy, joyful, excited, forgiving, etc.] about it. Give short answer.",
            'shadow': f"Make {{{str(response)}}} absolute negative, point out the worst in it and be [sad, angry, hateful, etc.] about it. Give short answer."
        }
        
        emotions = {}
        for key, query in emotion_queries.items():
            emotions[key] = query_engine.query(query)
            print(f"{key.title()}: {str(emotions[key])}")

        # Get external queries
        external_answer = process_external_queries(query_engine, response)
        
        # Reload core documents
        load_result = add_documents(core_documents, 5)
        if "Error" in load_result:
            yield history + [(message, f"Error loading content: {load_result}")]
            return

        # Final response synthesis
        pinial = query_engine.query(f"You are the pinial gland, emotion animal: {{{str(emotions['animal'])}}}, heart: {{{str(emotions['heart'])}}}, shadow: {{{str(emotions['shadow'])}}}, external: {{{str(external_answer)}}}, external answers: {{{str(external_answer)}}}. Give most relavant/suitable and correct answer to {{{message}}} Give short answer.")
        compared_result = query_engine.query(f"Main context: {{{str(responses['context'])}}}. Advanced reasoning: {{{str(pinial)}}}. Give answer to {{{message}}}.")

        yield history + [(message, str(compared_result))]
        
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AGI with RAG")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load Documents")

    load_output = gr.Textbox(label="Load Status")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question", interactive=True)
    clear = gr.Button("Clear")

    # Set up event handlers
    load_btn.click(handle_file_upload, inputs=[file_input], outputs=[load_output])
    msg.submit(stream_responses, inputs=[msg, chatbot], outputs = [chatbot])
    msg.submit(lambda: "", outputs=[msg])
    clear.click(lambda: None, None, chatbot, queue=False)

    #Launch the Gradio interface
    if __name__ == "__main__":
        demo.launch()