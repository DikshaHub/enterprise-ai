from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel 
from pypdf import PdfReader
from app.services.chunking import chunk_text, clean_text
from app.services.vector_store import add_chunks, search_chunks, build_context
from app.services.llm import generate_answer

app = FastAPI() 

class ChatRequest(BaseModel):
    question : str

@app.get("/")
def root():
    return {"message": "Enterprise AI Knowledge Platform Running!"} 

@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "question" : request.question,
        "message" : "Question received successfully!"
    }         

@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    text = clean_text(text)

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the PDF."
        )

    chunks = chunk_text(text)    
    add_chunks(file.filename, chunks)

    return {
        "filename": file.filename,
        "pages" : len(reader.pages),
        "chunk_count" : len(chunks),
        "chunks" : chunks
    }    


@app.get("/search")
def search_documents(query: str, n_results: int = 5):
    results = search_chunks(query, n_results)
    context = build_context(results)

    return {
        "query": query,
        "context": context,
        "results": results
    }

@app.get("/ask")
def ask_question(query: str, n_results: int = 5):
    results = search_chunks(query, n_results)

    if not results:
        return {
            "query": query,
            "answer": "I couldn't find the answer in the provided documents.",
            "sources": []
        }

    context = build_context(results)
    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer,
        "sources": results
    }    