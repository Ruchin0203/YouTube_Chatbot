# 🎥 YouTube RAG QA System (LangChain + OpenAI)

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** that allows you to **ask questions about a YouTube video** using its transcript. The system retrieves relevant transcript chunks and generates **grounded answers strictly from the video content**.

---

## ✨ What This Project Does

✅ Fetches YouTube video transcripts
✅ Splits long transcripts into chunks
✅ Embeds transcript chunks using OpenAI embeddings
✅ Stores embeddings in a FAISS vector database
✅ Retrieves relevant chunks for a user query
✅ Uses an LLM to answer questions **only from retrieved context**
✅ Prevents hallucinations by enforcing context-only answers

---

## 🧠 Architecture Overview

```
YouTube Video
     ↓
Transcript Extraction
     ↓
Text Chunking
     ↓
Embedding Generation (OpenAI)
     ↓
FAISS Vector Store
     ↓
Similarity Retrieval
     ↓
Prompt Augmentation
     ↓
LLM Answer (GPT-4o-mini)
```

---

## 🧱 Tech Stack

### Core Libraries

* **LangChain**
* **OpenAI API**
* **FAISS (Vector Database)**
* **YouTube Transcript API**

### Models Used

* **Embeddings:** `text-embedding-3-small`
* **LLM:** `gpt-4o-mini`

---

## 📦 Installation

### 1️⃣ Install Dependencies

```bash
pip install youtube-transcript-api \
            langchain-community \
            langchain-openai \
            faiss-cpu \
            tiktoken \
            python-dotenv
```

---

### 2️⃣ Set Environment Variables

⚠️ **DO NOT hardcode API keys in production**

```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
```

Or using `.env` (recommended):

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🚀 How It Works (Step-by-Step)

---

## 🔹 Step 1: Indexing (Document Ingestion)

### 1a️⃣ Fetch YouTube Transcript

```python
video_id = "Gfr50f6ZBvo"

transcript_list = YouTubeTranscriptApi.get_transcript(
    video_id,
    languages=["en"]
)

transcript = " ".join(chunk["text"] for chunk in transcript_list)
```

If captions are disabled, the app exits gracefully.

---

### 1b️⃣ Split Transcript into Chunks

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])
```

✔ Prevents context overflow
✔ Maintains semantic continuity

---

### 1c️⃣ Generate Embeddings & Store in FAISS

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(chunks, embeddings)
```

FAISS enables **fast similarity search** over transcript chunks.

---

## 🔹 Step 2: Retrieval

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

This retrieves the **top-k most relevant transcript chunks** for a question.

---

## 🔹 Step 3: Augmentation (Prompt Engineering)

The system **forces the LLM to answer only from retrieved context**.

```python
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)
```

🚫 No hallucinations
✅ Context-grounded answers

---

## 🔹 Step 4: Generation (LLM)

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)
```

Low temperature ensures **fact-based, stable answers**.

---

## 🔗 Full RAG Chain (Production-Style)

```python
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | StrOutputParser()
```

---

### 🧪 Example Queries

```python
main_chain.invoke("Who is Demis?")
```

```python
main_chain.invoke("Is nuclear fusion discussed in this video?")
```

```python
main_chain.invoke("Can you summarize the video?")
```

---

## 📌 Key Design Decisions

* 🔒 **Context-only answering** (no hallucination)
* ⚡ **FAISS for fast retrieval**
* 🧠 **OpenAI embeddings for semantic search**
* 🔁 **Composable LangChain runnables**
* 📚 **Chunk overlap to preserve meaning**

---

## 🛡️ Safety & Best Practices

* Never commit API keys
* Use `.env` files
* Limit chunk size to avoid token overflow
* Keep `temperature ≤ 0.3` for QA tasks

---

## 🚀 Possible Extensions

* ✅ Streamlit / Flask UI
* ✅ Multi-video indexing
* ✅ Persistent FAISS storage
* ✅ Timestamped answers
* ✅ Source citation per answer
* ✅ Whisper fallback if transcripts are disabled

---

## 📄 License

MIT License — free to use and modify.

---

## 🌟 Summary

This project demonstrates a **clean, real-world RAG pipeline** that turns YouTube videos into **queryable knowledge bases**, suitable for:

* AI tutors
* Video summarization
* Research assistants
* Internal knowledge tools

---

### 🧠 Built with LangChain + OpenAI

**Ask videos questions. Get grounded answers.**
