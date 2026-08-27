# YouTube Analyser Agent

An AI-powered application that allows users to interact with a YouTube video by asking questions about its content.

## Features

- Extracts transcripts from YouTube videos
- AI-powered question answering
- Semantic search using FAISS
- Retrieval-Augmented Generation (RAG)
- Hugging Face embeddings
- Supports OpenAI and Groq LLMs
- Interactive Streamlit interface
- Handles transcript and processing errors
- Automatically cleans up temporary transcript files

## 🏗️ How It Works

1. Enter a YouTube video URL.
2. Extract the video transcript.
3. Split the transcript into smaller chunks.
4. Generate embeddings using Hugging Face.
5. Store embeddings in a FAISS vector database.
6. Retrieve relevant content based on the user's question.
7. Generate a concise answer using the selected LLM.

## 🛠️ Tech Stack

- Python > 3.14
- Streamlit
- LangChain
- FAISS
- Hugging Face Embeddings
- OpenAI / Groq
- YouTube Transcript API
- Pytube

## ⚙️ Setup

Create a `.env` file in the same project directory and add the following values:

```env
OPENAI_API_KEY=<YOUR OPENAI API KEY>
GROQ_API_KEY=<YOUR GROQ API KEY>

chat_llm=groq
# chat_llm=open_api


```bash
git clone https://github.com/<your-username>/YouTube-Analyser-Agent.git
cd YouTube-Analyser-Agent
pip install -r requirements.txt
streamlit run main.py
```

### 🌐 Access the Application

After starting the Streamlit application, open your browser and visit:

**http://localhost:8501**

You can now access and interact with the application.