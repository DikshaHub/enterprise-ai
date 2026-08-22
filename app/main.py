from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel 
from pypdf import PdfReader
from app.services.chunking import chunk_text, clean_text

app = FastAPI() # create an instance of the FastAPI class

class ChatRequest(BaseModel):
    question : str

@app.get("/")
def root():
    return {"message": "Enterprise AI Knowledge Platform Running!"} 
        # return a JSON response with a message

@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "question" : request.question,
        "message" : "Question received successfully!"
    }         

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
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

    return {
        "filename": file.filename,
        "pages" : len(reader.pages),
        "chunk_count" : len(chunks),
        "chunks" : chunks
    }    


    

