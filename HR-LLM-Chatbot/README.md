# 🤖 HR LLM Chatbot (FastAPI + FAISS + LLM)

A **private HR chatbot** that answers HR-related questions using your company's Leave Policy, Travel Policy, and Performance Appraisal PDFs.

Built entirely with **open-source tools**: FastAPI, FAISS, SentenceTransformers, Hugging Face LLMs.


## Features

✅ Accepts natural language questions  
✅ Retrieves the most relevant HR policy sections  
✅ Generates accurate answers using a local LLM  
✅ Shows the source document and section for transparency  
✅ Runs locally — no data leaves your server  


## How to Run

### Step 1: Clone this Repo

```bash
git clone https://github.com/your-username/hr-policy-chatbot.git
```


### Step 2: Set Up Virtual Environment (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```


### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```


### Step 4: Place Your PDFs
Put your HR documents in the ./pdfs/ folder:
```bash
./pdfs/
├── Leave_Policy.pdf
├── Travel_and_Reimbursement_Policy.pdf
├── Performance_Appraisal_Policy.pdf
```


### Step 5: Set Your Hugging Face Token
In your app.py, set:
```bash
HF_TOKEN = "your_hf_token_here"
```
🔗 Get a token from: https://huggingface.co/settings/tokens


### Step 6: Start the App
```bash
python app.py
```
Then open http://localhost:8000 in your browser.


### 💬 Ask Your Bot
Try questions like:

"How many sick leaves are allowed?"/n"How is performance appraised?"/n"What's the travel reimbursement process?"


### API Example
Sample Request:
```bash
{
  "query": "How many days of maternity leave are allowed?"
}
```

Sample Response:
```bash
{
  "query": "...",
  "answer": "...",
  "sources": [
    {
      "id": "Leave Policy - 3. Maternity & Paternity Leave",
      "document": "Leave Policy",
      "section": "3. Maternity & Paternity Leave"
    }
  ]
}
```


### Architecture Diagram
```bash
                          ┌──────────────────────────────┐
                          │        HR Policy PDFs        │
                          │ (Leave, Travel, Appraisal)   │
                          └────────────┬─────────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────────┐
                          │        PDF Extractor         │
                          └────────────┬─────────────────┘
                                       ▼
                          ┌──────────────────────────────┐
                          │       Section Chunker        │
                          └────────────┬─────────────────┘
                                       ▼
                          ┌──────────────────────────────┐
                          │ Embeddings Generator         │
                          │ Model: all-MiniLM-L6-v2      │
                          └────────────┬─────────────────┘
                                       ▼
                          ┌──────────────────────────────┐
                          │ FAISS Vector Store           │
                          │ - Stores vector embeddings   │
                          │ - Enables top-k search       │
                          └────────────┬─────────────────┘
                                       ▲
                                       │
                   ┌───────────── User Query ─────────────┐
                   ▼                                      ▼
  ┌────────────────────────────┐         ┌────────────────────────────┐
  │ Chat UI (index.html)       │         │ FastAPI Backend (app.py)   │
  │ - Text input               │         │ - /ask endpoint            │
  │ - Displays answer/sources  │         │ - Embeds query             │
  └────────────────────────────┘         │ - Top-k retrieval (FAISS)  │
                                         │ - Prompt + context builder │
                                         └────────────┬───────────────┘
                                                      ▼
                          ┌──────────────────────────────┐
                          │ LLM Generator (HuggingFace)  │
                          │ Model: LLaMA 3.2-3B Instruct │
                          └────────────┬─────────────────┘
                                       ▼
                         ┌───────────────────────────────┐
                         │  Final Answer + Sources       │
                         └───────────────────────────────┘

```
