from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pdfplumber
import uvicorn
import tempfile, os
import base64
import faiss
import numpy as np
import ollama
from fastapi.middleware.cors import CORSMiddleware
# from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import OrderedDict
from pydantic import BaseModel
from typing import Optional, List
import json
from datetime import datetime
from functools import lru_cache

# Define feedback data model
class FeedbackData(BaseModel):
    rating: str  # "positive" or "negative"
    comment: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    timestamp: Optional[str] = None

# Store feedback data
feedback_store = []

app = FastAPI()

# ✅ Enable CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding and reranking models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ✅ FAISS index for fast vector search
index = faiss.IndexFlatL2(384)  # Dimension 384 for MiniLM embeddings
document_store = []  # Cache to store documents

# ✅ Cache for frequently accessed documents
cache = OrderedDict()
CACHE_SIZE = 100

def add_to_cache(key, value):
    if key in cache:
        cache.move_to_end(key)
    elif len(cache) >= CACHE_SIZE:
        cache.popitem(last=False)  # Remove least recently used item
    cache[key] = value

def process_document(file_bytes: bytes) -> list[Document]:
    """Processes an uploaded PDF file, extracts text, and splits it into smaller chunks."""
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(temp_fd, "wb") as temp_file:
            temp_file.write(file_bytes)
        
        with pdfplumber.open(temp_path) as pdf:
            # Process pages in parallel using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                text = "\n".join(executor.map(lambda page: page.extract_text() or "", pdf.pages))

    finally:
        os.unlink(temp_path)
    
    # Optimize chunk size and overlap for better performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size for better context
        chunk_overlap=200,  # Increased overlap for better continuity
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    doc = Document(page_content=text)
    return text_splitter.split_documents([doc])

# Add caching for embeddings
@lru_cache(maxsize=1000)
def get_embedding(text: str):
    """Cache embeddings to avoid recomputing."""
    return embedding_model.encode([text])[0]

def add_to_vector_store(all_splits: list[Document]):
    """Adds document splits to the FAISS index for fast retrieval."""
    global document_store, index
    
    # Process embeddings in parallel
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        texts = [split.page_content for split in all_splits]
        embeddings = list(executor.map(get_embedding, texts))
    
    # Add embeddings in batch
    index.add(np.array(embeddings, dtype=np.float32))
    document_store.extend(texts)
    return {"message": "Data added to the FAISS vector store!"}

def query_expansion(prompt: str) -> str:
    """Expands queries dynamically using synonyms or related terms."""
    # Simple expansion example (can use NLP models for more advanced expansion)
    expansions = {"AI": "artificial intelligence", "NLP": "natural language processing"}
    words = prompt.split()
    expanded_query = " ".join([expansions.get(word, word) for word in words])
    return expanded_query

def query_vector_store(prompt: str, n_results: int = 10):
    """Queries FAISS vector store and returns relevant documents."""
    global index, document_store
    
    if not document_store:
        return []
    
    # Use cached embedding
    query_embedding = get_embedding(query_expansion(prompt))
    n_results = min(n_results, len(document_store))
    
    # Use GPU if available for faster search
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = gpu_index.search(np.array([query_embedding], dtype=np.float32), n_results)
        del gpu_index
    else:
        D, I = index.search(np.array([query_embedding], dtype=np.float32), n_results)
    
    results = []
    for idx in I[0]:
        if 0 <= idx < len(document_store):
            results.append(document_store[idx])
    
    return results

# Add caching for cross-encoder predictions
@lru_cache(maxsize=1000)
def get_cross_encoder_score(prompt: str, doc: str):
    """Cache cross-encoder scores to avoid recomputing."""
    return cross_encoder.predict([(prompt, doc)])[0]

def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model."""
    # Process scores in parallel
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        scores = list(executor.map(lambda doc: get_cross_encoder_score(prompt, doc), documents))
    
    ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    relevant_text = " ".join([doc[0] for doc in ranked_results[:3]])
    return relevant_text, [i for i, _ in enumerate(ranked_results[:3])]

# Add caching for LLM responses
@lru_cache(maxsize=100)
def get_cached_llm_response(context: str, prompt: str) -> str:
    """Cache LLM responses for identical queries."""
    system_prompt = """You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text."""

    response = ollama.chat(
        model="gemma3:12b",
        stream=False,  # Use non-streaming for cached responses
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"},
        ],
    )
    return response["message"]["content"]

def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response."""
    # Try to get cached response first
    try:
        cached_response = get_cached_llm_response(context, prompt)
        yield cached_response
        return
    except:
        pass
    
    # If no cache, use streaming
    system_prompt = """You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text."""

    response = ollama.chat(
        model="gemma3:12b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"},
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"error": "Invalid file type"})
    
    contents = await file.read()
    splits = process_document(contents)
    result = add_to_vector_store(splits)
    return result

@app.post("/ask")
async def ask_question(prompt: str = Form(...)):
    cache_key = f"ask:{prompt}"
    if cache_key in cache:
        return cache[cache_key]
    
    # Add timeout to prevent long-running queries
    from asyncio import TimeoutError
    try:
        results = query_vector_store(prompt)
        if not results:
            return {"response": "No relevant documents found.", "retrieved_documents": [], "relevant_ids": []}
        
        relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, results)
        response_chunks = []
        
        # Use asyncio to handle streaming with timeout
        import asyncio
        async def collect_chunks():
            for chunk in call_llm(context=relevant_text, prompt=prompt):
                response_chunks.append(chunk)
        
        await asyncio.wait_for(collect_chunks(), timeout=30.0)  # 30 second timeout
        
        response_text = "".join(response_chunks)
        final_response = {
            "response": response_text,
            "retrieved_documents": results,
            "relevant_ids": relevant_text_ids,
        }
        add_to_cache(cache_key, final_response)
        return final_response
    except TimeoutError:
        return {"response": "Request timed out. Please try again.", "retrieved_documents": [], "relevant_ids": []}
    except Exception as e:
        return {"response": f"An error occurred: {str(e)}", "retrieved_documents": [], "relevant_ids": []}

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackData):
    """Store feedback for RLHF training."""
    try:
        # Add timestamp if not provided
        if not feedback.timestamp:
            feedback.timestamp = datetime.now().isoformat()
        
        # Store feedback
        feedback_store.append(feedback.model_dump())
        
        # Save feedback to file for persistence
        with open("feedback_data.json", "a") as f:
            json.dump(feedback.model_dump(), f)
            f.write("\n")
        
        return {"message": "Feedback received successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to store feedback: {str(e)}"}
        )

@app.get("/feedback")
async def get_feedback():
    """Retrieve stored feedback data."""
    return feedback_store

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
