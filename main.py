"""
Script RAG avec :
 • Embeddings Sentence‑Transformers + index FAISS
 • Choix du modèle FLAN‑T5 (base / large)
 • Interface console ou Gradio
 • Sauvegarde des Q/R et affichage des sources
Exécution :  python main.py
"""
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from datetime import datetime

# ---------- Couleurs console ----------
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class DummyColor:  # fallback sans couleur
        GREEN = ""
        RESET_ALL = ""
    Fore = DummyColor()
    Style = DummyColor()

# ---------- Gradio ----------
try:
    import gradio as gr
except ImportError:
    gr = None  # interface web désactivée si Gradio absent

# ---------- 1. Chargement documents ----------
def load_documents(folder="data"):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    docs.append(txt)
    return docs

def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

docs = load_documents()
chunks = []
for d in docs:
    chunks.extend(split_paragraphs(d))
print(f"👉  Fragments chargés : {len(chunks)}")

# ---------- 2. Embeddings & index FAISS ----------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print(f"✅  Index FAISS prêt : {index.ntotal} vecteurs\n")

# ---------- 3. Choix du modèle ----------
print("Quel modèle veux-tu utiliser ?")
print("1. flan-t5-base (rapide)")
print("2. flan-t5-large (plus puissant, plus lent)")
choice = input("Entrez 1 ou 2 : ").strip()
model_name = "google/flan-t5-large" if choice == "2" else "google/flan-t5-base"
print(f"🔍 Modèle sélectionné : {model_name.split('/')[-1]}")

generator = pipeline("text2text-generation", model=model_name, tokenizer=model_name)

# ---------- 4. Fonctions RAG ----------
def retrieve(query, k=3):
    q_emb = embedder.encode([query])
    _, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0] if i != -1]

def answer(question):
    ctx_list = retrieve(question, k=3)
    context = "\n".join(ctx_list)
    prompt = (
        "Réponds à la question en te basant sur le contexte suivant :\n"
        f"{context}\nQuestion : {question}"
    )
    out = generator(prompt, max_new_tokens=100)
    return out[0]["generated_text"], ctx_list

def log_qa(question, rep, logfile="log_qr.txt"):
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"📅 {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Question : {question}\nRéponse : {rep}\n{'-'*40}\n")

# ---------- 5. Interfaces ----------
def format_sources(srcs):
    lines = []
    for i, src in enumerate(srcs, 1):
        clean = src[:250].replace("\n", " ")
        ellipsis = "..." if len(src) > 250 else ""
        lines.append(f"[{i}] {clean}{ellipsis}")
    return "\n".join(lines)

def gradio_interface(question):
    rep, sources = answer(question)
    result = "📚 Fragments utilisés :\n" + format_sources(sources)
    result += f"\n\n💡 Réponse : {rep}"
    log_qa(question, rep)
    return result

# ---------- 6. Lancement ----------
if __name__ == "__main__":
    print("\n💻 Modes disponibles :")
    print("1. Terminal (console)")
    if gr is not None:
        print("2. Interface Web (Gradio)")
    mode = input("Choisissez le mode (1 ou 2) : ").strip()

    if mode == "2" and gr is not None:
        print("🌐 Lancement de Gradio...")
        gr.Interface(
            fn=gradio_interface,
            inputs="text",
            outputs="text",
            title="Système RAG Open Source",
            description="Posez une question sur vos documents"
        ).launch()
    else:
        print("\nBienvenue dans le système RAG 🚀 (tapez exit pour quitter)\n")
        while True:
            q = input("❓  Votre question : ").strip()
            if q.lower() in {"exit", "quit"}:
                print("👋 Fin de session")
                break
            rep, sources = answer(q)
            print("\n📚 Fragments utilisés :")
            print(format_sources(sources))
            print(Fore.GREEN + "\n💡  Réponse :", rep + Style.RESET_ALL)
            log_qa(q, rep)
