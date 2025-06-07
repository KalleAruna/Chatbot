from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn, nest_asyncio
import os
import re
import fitz
import json
import faiss
import torch
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# ============ CONFIGURATION ============
PDF_FOLDER = "./pdfs"
JSON_CHUNK_OUTPUT = "dynamic_policy_chunks.json"
FAISS_INDEX_OUTPUT = "faiss_policy_index.index"
FAISS_META_OUTPUT = "faiss_policy_metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
HF_TOKEN = ""

# ============ PREPROCESS PDFs ============
def extract_text_from_pdf(pdf_path: str) -> str:
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def get_section_chunks(text: str) -> List[Dict[str, str]]:
    pattern = re.compile(r'(?<=\n)(\d+(\.\d+)?\. .+?)(?=\n\d+(\.\d+)?\. |\Z)', re.DOTALL)
    matches = pattern.findall(text)
    chunks = []
    for match in matches:
        full_match = match[0].strip()
        lines = full_match.split("\n", 1)
        section_header = lines[0].strip()
        section_body = lines[1].strip() if len(lines) > 1 else ""
        chunks.append({"section": section_header, "content": section_body})
    return chunks

def generate_chunk_id(base: str, section: str) -> str:
    return f"{base.strip()} - {section.strip()}"

def process_document(pdf_path: str) -> List[Dict]:
    base_name = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ")
    text = extract_text_from_pdf(pdf_path)
    sections = get_section_chunks(text)
    return [{
        "id": generate_chunk_id(base_name, sec["section"]),
        "text": sec["content"],
        "metadata": {
            "document_title": base_name,
            "section_heading": sec["section"]
        }
    } for sec in sections]

def process_all_pdfs(folder_path: str) -> List[Dict]:
    chunks = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            chunks.extend(process_document(os.path.join(folder_path, file)))
    return chunks

# ============ SETUP FASTAPI ============
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    global chunks, texts, metadata, ids, model, index, tokenizer, llm, generator

    print("Loading PDF chunks...")
    chunks = process_all_pdfs(PDF_FOLDER)
    with open(JSON_CHUNK_OUTPUT, "w") as f:
        json.dump(chunks, f, indent=2)

    texts = [chunk["text"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    print("Computing embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, FAISS_INDEX_OUTPUT)
    with open(FAISS_META_OUTPUT, "w") as f:
        json.dump({"ids": ids, "metadata": metadata}, f, indent=2)

    print("Loading LLaMA model...")
    login(token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, torch_dtype=torch.float32, device_map={"": "cpu"})
    generator = pipeline("text-generation", model=llm, tokenizer=tokenizer)

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html") as f:
        return f.read()

@app.post("/ask")
def ask(request: QueryRequest):
    query = request.query
    top_k = 3
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)

    retrieved_docs = [{
        "id": ids[i],
        "metadata": metadata[i],
        "text": texts[i]
    } for i in I[0]]

    context = "\n\n".join([f"{doc['id']}:\n{doc['text']}" for doc in retrieved_docs])
    prompt = f"""Answer the following question based only on the context provided.\n
### Context:\n{context}\n
### Question:\n{query}\n
### Answer:"""

    output = generator(prompt, max_new_tokens=512, do_sample=False, top_k=3, return_full_text=False)
    answer = output[0]['generated_text'].strip()

    # return {"query": query, "answer": answer}
    return {
    "query": query,
    "answer": answer,
    "sources": [
        {
            "id": doc["id"],
            "document": doc["metadata"]["document_title"],
            "section": doc["metadata"]["section_heading"]
        }
        for doc in retrieved_docs
    ]
}

# ============ RUN ============
if __name__ == "__main__":
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
