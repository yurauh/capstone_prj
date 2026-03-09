## RAG Chatbot (PDF + Audio)

This project implements a simple Retrieval-Augmented Generation (RAG) chatbot in Python that:

- **Ingests PDFs and audio recordings**
- **Transcribes audio to text using a local Whisper model**
- **Chunks text into semantically reasonable pieces**
- **Embeds chunks and stores them in a local ChromaDB vector store**
- **Retrieves relevant chunks for a user question and calls a local LLM to answer**

### 1. Setup

1. **Create and activate a virtual environment (recommended)**:

```bash
python -m venv .venv
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set your Hugging Face API key**:

The code expects the environment variable `HUGGINGFACE_API_KEY` to be set:

```bash
export HUGGINGFACE_API_KEY="your_hf_token_here"
```

### 2. Prepare Your Data

Create the data folders:

```bash
mkdir -p data/pdfs data/audio
```

- **Place PDF files** (e.g. `kb1.pdf`, `manual.pdf`) into `data/pdfs`.
- **Place audio files** (e.g. `notes.wav`, `lecture.mp3`) into `data/audio`.

Supported audio extensions in the default script: `wav`, `mp3`, `m4a`.

4. **Install and run a local LLM with Ollama**

- Install Ollama from [`https://ollama.com`](https://ollama.com).
- Pull a model (for example `llama3`):

```bash
ollama pull llama3
```

- Start the Ollama service (if not already running):

```bash
ollama serve
```

The script assumes:

- API URL: `http://localhost:11434/api/generate`
- Model name: `llama3`

You can change these in `rag_chatbot.py` by editing `OLLAMA_API_URL` and `OLLAMA_MODEL_NAME`.

### 3. How the Pipeline Works

The main script is `rag_chatbot.py`:

- **PDF ingestion**: `load_pdf_text` reads each PDF and extracts the text with `pypdf`.
- **Audio ingestion**: `transcribe_audio_to_text` runs a local Whisper model (via `openai-whisper`) to generate the transcription. This requires `ffmpeg` installed on your system.
- **Chunking**: `split_text_into_chunks` splits long texts into overlapping chunks by word count (approximate token limit).
- **Embedding + storage**:
  - `embed_texts` uses a Hugging Face embedding model (`sentence-transformers/all-MiniLM-L6-v2`) through the Inference API.
  - Chunks are written to a **ChromaDB** persistent collection (`chroma_db` folder).
- **Retrieval + generation**:
  - `query_vector_store` embeds the question and retrieves top-k similar chunks.
  - `generate_answer_from_context` sends the question + retrieved context to the local Ollama model (default: `llama3`) to generate the final answer.

### 4. Run the Chatbot

After preparing data and setting `HUGGINGFACE_API_KEY`, run:

```bash
python rag_chatbot.py
```

The script will:

- Scan `data/pdfs` and `data/audio` for files.
- Build the RAG index in ChromaDB (or reuse the existing persistent store).
- Start an interactive **CLI chat**:
  - Type your question and press Enter.
  - Type `exit` or `quit` to stop.

## Agentic RAG System (Reason -> Act -> Reflect)

This repository now also includes an **agentic** workflow in `agentic_rag.py`.

It extends the previous RAG chatbot with:

- **Data contextualization**: reuses your PDF + audio ingestion and vector index.
- **Reasoning loop**: planner chooses the next tool action for each step.
- **Tool calling**:
  - `search_knowledge_base` (vector retrieval),
  - `inspect_project_file` (local file inspection),
  - `inspect_chat_logs` (inspect previous chat traces).
- **Self-reflection**: after each tool call, the agent critiques whether evidence is sufficient and suggests the next step.
- **Evaluation**: computes simple `accuracy`, `relevance`, `clarity`, and `overall` metrics.

### Run Agentic Modes

1. Build or refresh the index:

```bash
python agentic_rag.py --mode build_index
```

2. Run interactive agentic chat:

```bash
python agentic_rag.py --mode chat
```

3. Run evaluation using a JSON dataset:

```bash
python agentic_rag.py --mode evaluate --eval-file evaluation_dataset.json
```

Evaluation file format:

```json
[
  {
    "question": "What are the key takeaways?",
    "expected_answer": "Brief expected content used to score overlap and quality."
  }
]
```

