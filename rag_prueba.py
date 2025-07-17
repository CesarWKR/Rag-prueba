# RAG con PDF en inglés y preguntas en español
# =============================================
# Requisitos: pip install sentence-transformers faiss-cpu openai PyPDF2

import faiss
from openai import OpenAI
import pickle
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Cargar variables de entorno
load_dotenv()

# --- CONFIGURACIÓN ---
EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v1"
PDF_PATH = "docs/gg243376.pdf"  # Importar PDF en inglés
INDEX_PATH = "rag_index.index"
CHUNKS_PATH = "rag_chunks.pkl"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF no encontrado en {PDF_PATH}. ¿Lo colocaste en la carpeta 'docs'?")


# --- 1. Ingesta de documento ---
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- 2. Preprocesamiento y embeddings ---
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def create_faiss_index(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def save_index(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_chunks():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# --- 3. Recuperación semántica ---
def search_chunks(query, index, chunks, top_k=3):
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_embed = model.encode([query])
    D, I = index.search(q_embed, top_k)
    return [chunks[i] for i in I[0]]

# --- 4. Generación con OpenAI ---
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
Eres un asistente técnico. A continuación se proporciona un contexto técnico en inglés extraído de un documento.
Responde la pregunta en español basándote únicamente en la información proporcionada.

Contexto:
{context}

Pregunta: {query}
Respuesta:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, # Baja temperatura para respuestas más precisas
        max_tokens=400  # Limitar tokens para respuestas concisas
    )
    return response.choices[0].message.content.strip()

# --- 5. Evaluación ---
def evaluate(questions):
    index, chunks = load_index_and_chunks()
    for q in questions:
        top_chunks = search_chunks(q, index, chunks)
        answer = generate_answer(q, top_chunks)
        print("\n❓ Pregunta:", q)
        print("🧠 Respuesta:", answer)

# --- MAIN (primer uso) ---
if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):  # Si no existe el índice, crearlo para no reindexar cada vez que se ejecuta
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text, chunk_size=500)
        index, chunks = create_faiss_index(chunks)
        save_index(index, chunks)

    preguntas = [
        "¿Qué es el protocolo TCP/IP?",
        "¿Cuáles son las capas del modelo TCP/IP?",
        "¿Qué diferencia hay entre TCP y UDP?"
    ]
    evaluate(preguntas)