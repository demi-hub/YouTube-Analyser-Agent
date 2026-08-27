import os
import atexit
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

from pytube import YouTube
from youtube_transcript_api import(
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
transcript_file_name = "transcript.txt"

def cleanup_transcript():
    try:
        if os.path.exists(transcript_file_name):
            with open(transcript_file_name, "w") as file:
                file.truncate(0)
    except Exception as err:
        print(f"Error while clearing transcript file: {err}")
        
atexit.register(cleanup_transcript)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def get_llm():
    try:
        chat_llm = os.environ.get("chat_llm")
        openapi_llm = ChatOpenAI(model="gpt-4o-mini")
        grok_llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

        if chat_llm == "open_api":
            return openapi_llm
        elif chat_llm == "groq":
            return grok_llm
        else:
            raise ValueError(
                "Invalid LLM name. "
            )
    except Exception as err:
        st.error(f"Failed to initialize LLM: {err}")
        return None
        
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as err:
        st.error(f"Failed to initialize embeddings: {err}")
        return None

def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        ts_list = YouTubeTranscriptApi().list(video_id)
        ts_data = ts_list.find_transcript(["en"])
        youtube_text = "\n\n".join(
            ts.text for ts in ts_data.fetch()
        )
        return youtube_text
    except TranscriptsDisabled:
        st.error("Transcript Disabled")
    except NoTranscriptFound:
        st.error("No Transcript Found")
    except VideoUnavailable:
        st.error("Video Unavailable")
    except Exception as err:
        print(f"Unexpected Error {str(err)}")

def save_ts_to_file(text, filename=transcript_file_name):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
            return True
    except Exception as err:
        st.error(f"Failed to save transcript: {err}")
        return False

def create_youtube_retrieval_chain(llm, embeddings):
    try:
        loader = TextLoader(transcript_file_name, encoding="utf-8")
        documents = loader.load()
        
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError(
                    "Transcript could not be split into chunks."
                )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever()
        
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
        return retrieval_chain
    
    except Exception as err:
        st.error(
            f"Failed to create retrieval chain: {err}"
        )
        return None
    
def main():
    try:
        llm = get_llm()
        if llm is None:
            st.stop()

        embeddings = get_embeddings()
        if embeddings is None:
            st.stop()
        
        # streamlit UI
        st.title("AI Youtube Analyser")
        st.write("Ask questions from your personalized youtube link")

        video_url = st.text_input("Please Enter Youtube URL: ")

        if st.button("process_video"):
            if not video_url.strip():
                st.warning(
                    "Please enter a valid YouTube URL."
                )
            else:
                with st.spinner(
                    "Fetching and processing transcript..."
                ):
                    transcripts = get_youtube_transcript(video_url)
                    if transcripts:
                        saved = save_ts_to_file(transcripts)
                        if saved:
                            retrieval_chain = (
                                create_youtube_retrieval_chain(
                                    llm,
                                    embeddings
                                )
                            )
                            
                            if retrieval_chain:
                                st.session_state.retrieval_chain = (
                                    retrieval_chain
                                )

                                st.success(
                                    "Transcript processed successfully. "
                                    "You can ask questions now."
                                )
                    
        if "retrieval_chain" in st.session_state:
            user_question = st.text_input("Ask a question")
            if user_question.strip():
                try:
                    response = st.session_state.retrieval_chain.invoke(
                        {"input": user_question}
                    )
                    st.write("**Answer:**", response.get("answer", "I don't know."))
                except Exception as err:
                    st.error(f"Failed to generate answer: {err}")
    
    except Exception as err:
        st.error(
            f"Unexpected application error: {err}"
        )

if __name__ == "__main__":
    main()
