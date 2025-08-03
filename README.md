# 📚 RAG Document Q&A System

An interactive Streamlit app for uploading documents and asking questions using **Retrieval-Augmented Generation (RAG)**. It processes your documents into chunks, stores them in a vector store, and uses an LLM to answer queries based on the content.

---

## 🚀 Features

- 📤 Upload multiple document formats: PDF, TXT, CSV, DOCX, DOC
- 🔍 Automatic document chunking and vector embedding
- 🧠 LLM backend options:
  - `local`: Minimal logic-based fallback
  - `ollama`: Local LLMs like Llama2 via [Ollama](https://ollama.ai)
  - `huggingface`: Hosted transformer models via Hugging Face API
- 💬 Natural language Q&A from your files
- 📄 Source documents shown with each answer
- ⚙️ Configurable settings for chunking, creativity, and top-k retrieval

---

## 📂 Project Structure

```
RAG-QA-App/
├── app.py              # Main Streamlit app
├── rag_pipeline.py     # Core RAG pipeline classes (RAGPipeline, RAGConfig, DocumentProcessor)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🛠 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Whateverdoes/RAG.git
cd rag-doc-qa
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🧱 Tech Stack

- [Streamlit](https://streamlit.io/) – UI framework
- [LangChain](https://www.langchain.com/) – Chunking, embedding, querying
- [Ollama](https://ollama.ai) – Run local LLMs (LLaMA2, Mistral, etc.)
- [Hugging Face Transformers](https://huggingface.co/) – Hosted models
- [FAISS / ChromaDB](https://github.com/facebookresearch/faiss) – Vector DBs

---

## 🔌 LLM Backend Options

### ✅ Local
Basic logic only; does not use any LLM.

### 🦙 Ollama (Recommended)

1. Download from: https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull llama2
   ```
3. Select `ollama` from sidebar

### 🤗 Hugging Face

1. Get a token: https://huggingface.co/settings/tokens
2. Enter token in sidebar
3. Choose `huggingface` as backend

---

## ⚙️ Config Options

| Setting         | Description                              |
|----------------|------------------------------------------|
| Chunk Size      | Characters per chunk                    |
| Chunk Overlap   | Overlap between chunks                  |
| Temperature     | Response randomness (0.0 = deterministic)|
| Top K           | Number of chunks retrieved              |

---

## 💬 Example Questions

- "Summarize the uploaded document."
- "What are the main points covered?"
- "List all mentioned authors."


---

## 🛡 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more information.

---

## 🤝 Contributing

Pull requests are welcome.  
For suggestions or bug reports, open an issue.

---
