import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import av
from typing import List
import tempfile
import os
from gtts import gTTS

# LangChain imports
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
import speech_recognition as sr
import soundfile as sf

class AudioProcessor:
    def __init__(self):
        self.chunks = []
        self.recording = False
        self.sample_rate = 16000

    def process_audio(self, frame):
        if self.recording:
            sound = frame.to_ndarray()
            self.chunks.append(sound)
        return frame

def create_audio_processor():
    return AudioProcessor()

def transcribe_audio(audio_data: np.ndarray) -> str:
    """Transcribe audio using SpeechRecognition"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, 16000)
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp_file.name) as source:
                audio = recognizer.record(source)
                return recognizer.recognize_google(audio)
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""
    finally:
        if 'tmp_file' in locals():
            os.unlink(tmp_file.name)

def setup_agent():
    try:
        # Initialize LLM
        llm = ChatGroq(
            model_name="gemma2-9b-it",
            api_key=st.secrets["GROQ_API_KEY"]
        )

        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.7
        )

        # Load CSV data
        csv = CSVLoader("table_data.csv")
        csv_data = csv.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(csv_data)

        # Create vector store
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever()

        # Create tools
        tool1 = create_retriever_tool(
            retriever,
            "reservation_data_tool",
            "has the table data for the purpose of making reservations"
        )

        def say_hello(customer_input):
            if customer_input == "Hello" or customer_input == "Hey":
                return "Hello Welcome to LeChateu how can I help you today?"
            return None

        tool2 = Tool.from_function(
            func=say_hello,
            name="say_hello_tool",
            description="use this tool to greet the customer after the customer has greeted you"
        )

        tools = [tool1, tool2]

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "if user specifies table number and time:present the user with the available tables and their locations available during the specified time"),
            ("system", "if user greets:greet the user back using the say_hello_tool"),
            ("system", "if user makes choice:confirm the reservation and conclude the conversation"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        agent = create_openai_tools_agent(llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            llm=llm,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=15
        )
    except Exception as e:
        st.error(f"Agent setup error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="LeChateau Reservation",
        page_icon="🍽️",
        layout="wide"
    )

    st.title("🎙️ LeChateau Voice Reservation System")

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = setup_agent()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Voice Interface")
        
        # Audio recording interface
        webrtc_ctx = webrtc_streamer(
            key="voice-recorder",
            media_stream_constraints={
                "audio": True,
                "video": False
            },
            audio_processor_factory=create_audio_processor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )

        if webrtc_ctx.state.playing:
            st.info("🎙️ Recording in progress...")
            if hasattr(webrtc_ctx, 'audio_processor'):
                webrtc_ctx.audio_processor.recording = True

        if st.button("Process Recording"):
            if hasattr(webrtc_ctx, 'audio_processor') and webrtc_ctx.audio_processor.chunks:
                with st.spinner("Processing your request..."):
                    # Process audio
                    audio_data = np.concatenate(webrtc_ctx.audio_processor.chunks)
                    transcript = transcribe_audio(audio_data)
                    
                    if transcript:
                        st.info(f"You said: {transcript}")
                        
                        # Get AI response
                        response = st.session_state.agent.invoke({
                            "input": transcript,
                            "chat_history": st.session_state.chat_history
                        })
                        
                        # Update chat history
                        st.session_state.chat_history.extend([
                            response['input'],
                            response['output']
                        ])
                        
                        # Generate and play audio response
                        tts = gTTS(response['output'], lang='en')
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                            tts.save(temp_audio.name)
                            st.audio(temp_audio.name)
                            os.unlink(temp_audio.name)
                    
                    # Clear recording
                    webrtc_ctx.audio_processor.chunks = []

    with col2:
        st.subheader("Conversation History")
        for message in st.session_state.chat_history:
            st.text(message)

if __name__ == "__main__":
    main()
