import chromadb
from app.services.embedding import create_embedding

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="documents")

def add_document(
    document_id : str,
    text: str,
    embedding,
    metadata: dict
):
    
    collection.add(
        ids= [document_id],
        documents= [text],
        embeddings= [embedding.tolist()],
        metadatas= [metadata]
    )

def delete_document(document_id: str):
    collection.delete(ids=[document_id])    

def add_chunks(filename:str,chunks: list[str]):
    for index, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)

        add_document(
            document_id = f"{filename}_chunk_{index}",
            text = chunk,
            embedding = embedding,
            metadata = {
                "filename": filename,
                "chunk_index": index
            }
        )       

def search_chunks(query: str, n_results: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    return [
        {
            "text": document,
            "distance": distance,
            "metadata": metadata

        }
        for document, distance, metadata in zip(documents, distances, metadatas)
        if distance < 0.7
    ]

def build_context(results):
    return "\n\n".join(
        result["text"]
        for result in results
    )