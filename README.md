# 🤖 FLAN-T5 Gradio RAG – Retrieval-Augmented Generation with FLAN-T5

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using the `google/flan-t5` model for answering user questions based on uploaded documents. It uses **sentence-transformers** for embedding text, **FAISS** for similarity search, and a **Gradio** web interface for interaction.

---

## 🚀 Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/your-username/flan-t5-gradio.git
cd flan-t5-gradio
```

## Install the dependencies

pip install -r requirements.txt

## Install Gradio for 
pip install gradio


## Project Structure

├── main.py          # Main file that runs the RAG pipeline and Gradio app
├── data/            # Folder containing .txt documents uploaded by user
├── .gradio/         # Gradio-related cache/config (auto-generated)
├── requirements.txt # Python dependencies


## Uploading Documents
To provide source content for answering questions, place your .txt files inside the data/ directory:

/data
 ├── doc1.txt
 └── notes.txt

## 💬 Running the App
## Launch the interface using:

python main.py
