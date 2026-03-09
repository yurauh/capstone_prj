import os
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from tqdm import tqdm
import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
import whisper
import requests


load_dotenv()


HF_API_KEY_ENV: str = "HUGGINGFACE_API_KEY"
EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL_NAME: str = "base"
OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME: str = "llama3.2:latest"
CHROMA_DIR: str = "chroma_db"
COLLECTION_NAME: str = "rag_knowledge_base"
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 100
LOG_FILE_PATH: str = "chat_logs.txt"


hf_client: InferenceClient = InferenceClient(
    api_key=os.environ.get(HF_API_KEY_ENV),
)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


def ensure_api_key_is_set() -> None:
    if os.environ.get(HF_API_KEY_ENV):
        return
    raise RuntimeError(
        f"Environment variable {HF_API_KEY_ENV} is not set. "
        f"Set it before running this script."
    )


def get_whisper_model() -> whisper.Whisper:
    return whisper.load_model(WHISPER_MODEL_NAME)


def load_pdf_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    reader: PdfReader = PdfReader(pdf_path)
    pages_text: List[str] = []
    for page in reader.pages:
        page_text: str = page.extract_text() or ""
        cleaned_text: str = page_text.strip()
        if cleaned_text:
            pages_text.append(cleaned_text)
    return "\n\n".join(pages_text)


def transcribe_audio_to_text(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model = get_whisper_model()
    result = model.transcribe(audio_path)
    text: str = str(result.get("text", "")).strip()
    return text


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    if not text:
        return []
    words: List[str] = text.split()
    chunks: List[str] = []
    start: int = 0
    approximate_tokens_per_word: int = 4
    max_words: int = chunk_size // approximate_tokens_per_word
    overlap_words: int = chunk_overlap // approximate_tokens_per_word
    if max_words <= 0:
        raise ValueError("chunk_size is too small for the approximate token size.")
    if overlap_words >= max_words:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    while start < len(words):
        end: int = min(len(words), start + max_words)
        chunk_words: List[str] = words[start:end]
        chunk_text: str = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def create_chroma_client(persist_directory: str = CHROMA_DIR) -> ClientAPI:
    os.makedirs(persist_directory, exist_ok=True)
    chroma_client: ClientAPI = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )
    return chroma_client


def get_or_create_collection(chroma_client: ClientAPI, name: str) -> Any:
    return chroma_client.get_or_create_collection(name=name)


def embed_texts(texts: List[str]) -> List[List[float]]:
    ensure_api_key_is_set()
    if not texts:
        return []
    embeddings: List[List[float]] = []
    for text in texts:
        embedding_result = hf_client.feature_extraction(
            text,
            model=EMBEDDING_MODEL_ID,
        )
        if isinstance(embedding_result[0], list):
            flattened: List[float] = [float(value) for value in embedding_result[0]]
            embeddings.append(flattened)
        else:
            flattened_single: List[float] = [float(value) for value in embedding_result]
            embeddings.append(flattened_single)
    return embeddings


def build_document_chunks_from_sources(
    pdf_paths: List[str],
    audio_paths: List[str],
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    counter: int = 0
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        pdf_text: str = load_pdf_text(pdf_path)
        pdf_chunks: List[str] = split_text_into_chunks(pdf_text)
        for i, chunk_text in enumerate(pdf_chunks):
            chunk_id: str = f"pdf_{counter}_{i}"
            metadata: Dict[str, Any] = {
                "source": "pdf",
                "file_path": os.path.abspath(pdf_path),
                "chunk_index": i,
            }
            chunks.append(DocumentChunk(id=chunk_id, text=chunk_text, metadata=metadata))
        counter += 1
    for audio_path in tqdm(audio_paths, desc="Processing audio files"):
        transcription: str = transcribe_audio_to_text(audio_path)
        audio_chunks: List[str] = split_text_into_chunks(transcription)
        for i, chunk_text in enumerate(audio_chunks):
            chunk_id = f"audio_{counter}_{i}"
            metadata = {
                "source": "audio",
                "file_path": os.path.abspath(audio_path),
                "chunk_index": i,
            }
            chunks.append(DocumentChunk(id=chunk_id, text=chunk_text, metadata=metadata))
        counter += 1
    return chunks


def populate_vector_store(
    chroma_client: ClientAPI,
    chunks: List[DocumentChunk],
    collection_name: str = COLLECTION_NAME,
) -> None:
    if not chunks:
        raise ValueError("No chunks to add to the vector store.")
    collection = get_or_create_collection(chroma_client, collection_name)
    texts: List[str] = [chunk.text for chunk in chunks]
    metadatas: List[Dict[str, Any]] = [chunk.metadata for chunk in chunks]
    ids: List[str] = [chunk.id for chunk in chunks]
    print(f"Embedding and adding {len(chunks)} chunks to ChromaDB...")
    embeddings: List[List[float]] = embed_texts(texts)
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print("Vector store populated.")


def query_vector_store(
    chroma_client: ClientAPI,
    query: str,
    top_k: int = 10,
    collection_name: str = COLLECTION_NAME,
) -> List[DocumentChunk]:
    collection = get_or_create_collection(chroma_client, collection_name)
    query_embedding: List[List[float]] = embed_texts([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    documents: List[str] = results.get("documents", [[]])[0]
    metadatas_list: List[Dict[str, Any]] = results.get("metadatas", [[]])[0]
    ids_list: List[str] = results.get("ids", [[]])[0]
    chunks: List[DocumentChunk] = []
    for doc_id, doc_text, metadata in zip(ids_list, documents, metadatas_list):
        chunks.append(
            DocumentChunk(
                id=doc_id,
                text=doc_text,
                metadata=metadata,
            )
        )
    return chunks


def generate_answer_from_context(question: str, context_chunks: List[DocumentChunk]) -> str:
    if not context_chunks:
        raise ValueError("No context chunks provided to generate an answer.")
    context_text: str = "\n\n".join(
        [
            f"Source: {chunk.metadata.get('source')} ({chunk.metadata.get('file_path')}, chunk {chunk.metadata.get('chunk_index')})\n"
            f"{chunk.text}"
            for chunk in context_chunks
        ]
    )
    prompt: str = (
        "You are a helpful assistant that answers user questions based only on the "
        "provided context. If the answer is not contained in the context, say that "
        "you do not know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
    response.raise_for_status()
    data: Dict[str, Any] = response.json()
    generated: str = str(data.get("response", "")).strip()
    return generated


def build_rag_pipeline(
    pdf_paths: List[str],
    audio_paths: List[str],
) -> ClientAPI:
    chroma_client: ClientAPI = create_chroma_client()
    chunks: List[DocumentChunk] = build_document_chunks_from_sources(
        pdf_paths=pdf_paths,
        audio_paths=audio_paths,
    )
    populate_vector_store(chroma_client=chroma_client, chunks=chunks)
    return chroma_client


def run_cli_chat(chroma_client: ClientAPI) -> None:
    print("RAG Chatbot is ready. Type 'exit' to quit.")
    while True:
        question: str = input("\nYour question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            print("Please enter a question.")
            continue
        try:
            context_chunks: List[DocumentChunk] = query_vector_store(
                chroma_client=chroma_client,
                query=question,
                top_k=10,
            )
            if not context_chunks:
                print("No relevant context found in the knowledge base.")
                continue
            print("\nRetrieved context chunks:")
            for chunk in context_chunks:
                preview: str = chunk.text[:200].replace("\n", " ")
                print(
                    f"- {chunk.metadata.get('source')} | "
                    f"{chunk.metadata.get('file_path')} | "
                    f"chunk {chunk.metadata.get('chunk_index')}: "
                    f"{preview}..."
                )
            answer: str = generate_answer_from_context(
                question=question,
                context_chunks=context_chunks,
            )
            print(f"\nAnswer:\n{answer}")
            try:
                with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
                    log_file.write("=== Entry ===\n")
                    log_file.write(f"Question: {question}\n")
                    log_file.write(f"Answer: {answer}\n\n")
            except OSError as err:
                print(f"Warning: failed to write log file: {err}")
        except Exception as err:
            print(f"Error while generating answer: {err}")


def main() -> None:
    ensure_api_key_is_set()
    pdf_dir: str = os.path.join("data", "pdfs")
    audio_dir: str = os.path.join("data", "audio")
    pdf_paths: List[str] = []
    audio_paths: List[str] = []
    if os.path.isdir(pdf_dir):
        for name in os.listdir(pdf_dir):
            if name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(pdf_dir, name))
    if os.path.isdir(audio_dir):
        for name in os.listdir(audio_dir):
            lower: str = name.lower()
            if lower.endswith(".wav") or lower.endswith(".mp3") or lower.endswith(".m4a"):
                audio_paths.append(os.path.join(audio_dir, name))
    if not pdf_paths and not audio_paths:
        raise RuntimeError(
            "No PDF or audio files found.\n"
            "Place PDF files in 'data/pdfs' and audio files in 'data/audio', "
            "then run this script again."
        )
    print(f"Found {len(pdf_paths)} PDF(s) and {len(audio_paths)} audio file(s).")
    chroma_client: ClientAPI = build_rag_pipeline(
        pdf_paths=pdf_paths,
        audio_paths=audio_paths,
    )
    run_cli_chat(chroma_client=chroma_client)


if __name__ == "__main__":
    main()

