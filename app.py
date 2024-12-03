import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import av
from typing import List
from gtts import gTTS
import tempfile
import os
from google.cloud import speech
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
import json

# Custom luxury styling
LUXURY_STYLE = """
<style>
    .main { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); }
    
    .luxury-container {
        background: linear-gradient(145deg, rgba(26, 26, 26, 0.9) 0%, rgba(45, 45, 45, 0.9) 100%);
        border: 1px solid #B8860B;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(184, 134, 11, 0.1);
    }
    
    .luxury-header {
        background: linear-gradient(to right, #B8860B, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Cinzel', serif;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
    }
    
    .luxury-subheader {
        color: #DAA520;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.5rem;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }
    
    .stButton button {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        color: #DAA520;
        border: 2px solid #B8860B;
        border-radius: 50px;
        padding: 1rem 2rem;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        font-family: 'Cormorant Garamond', serif;
        color: #DAA520;
    }
    
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        background: #DAA520;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }

    .response-box {
        background: rgba(184, 134, 11, 0.1);
        border-left: 3px solid #B8860B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
"""

# Sample restaurant data
RESTAURANT_DATA = """
table_number,location,capacity,reservation_times
1,Window,4,18:00-20:00
2,Garden,2,19:00-21:00
3,Main Floor,6,17:00-19:00
4,Terrace,2,18:30-20:30
5,Bar Area,4,20:00-22:00
"""

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

def setup_langchain():
    """Initialize LangChain components"""
    # Initialize LLM
    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="mixtral-8x7b-32768"
    )
    
    # Create temporary CSV file with restaurant data
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(RESTAURANT_DATA)
        temp_csv_path = f.name

    # Load and process restaurant data
    loader = CSVLoader(temp_csv_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(docs, llm)
    
    # Create tools
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        "restaurant_info",
        "Search for restaurant table information and availability"
    )
    
    greeting_tool = Tool(
        name="greeting",
        func=lambda x: "Welcome to LeChateau! How may I assist you with your reservation today?",
        description="Use this to greet the customer"
    )
    
    tools = [retriever_tool, greeting_tool]
    
    # Create prompt
    prompt = PromptTemplate(
        template="""You are an elegant AI concierge at LeChateau restaurant. 
        Always maintain a formal and sophisticated tone.
        
        When handling reservations:
        1. Greet the guest warmly
        2. Help them find an available table based on their preferences
        3. Confirm the details of their reservation
        
        Available tools: {tools}
        
        Current conversation:
        {input}
        
        Think carefully about your response:
        {agent_scratchpad}
        """,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={"tools": str(tools)}
    )
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

def transcribe_audio(audio_data: np.ndarray) -> str:
    """Transcribe audio using Google Cloud Speech-to-Text"""
    client = speech.SpeechClient()
    
    # Convert audio to proper format
    audio = speech.RecognitionAudio(content=audio_data.tobytes())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    
    response = client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else ""

def text_to_speech(text: str) -> str:
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            return temp_file.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="LeChateau Voice Concierge",
        page_icon="👑",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = setup_langchain()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Inject custom styling
    st.markdown(LUXURY_STYLE, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="luxury-header">LeChateau Voice Concierge</div>', unsafe_allow_html=True)
    st.markdown('<div class="luxury-subheader">Your Personal Reservation Assistant</div>', unsafe_allow_html=True)

    # Main content columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="luxury-container">', unsafe_allow_html=True)
        
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
            },
            translations={
                "start": "🎙️ Start Recording",
                "stop": "⏹️ Stop Recording"
            }
        )

        # Recording status indicator
        if webrtc_ctx.state.playing:
            st.markdown(
                '<div style="text-align: center; color: #DAA520;">'
                '<div class="status-indicator"></div>'
                'Recording in progress...</div>',
                unsafe_allow_html=True
            )
            if hasattr(webrtc_ctx, 'audio_processor'):
                webrtc_ctx.audio_processor.recording = True

        # Process recording button
        if st.button("✨ Process Recording", key="process"):
            if hasattr(webrtc_ctx, 'audio_processor') and webrtc_ctx.audio_processor.chunks:
                with st.spinner("Processing your request..."):
                    # Combine audio chunks
                    audio_data = np.concatenate(webrtc_ctx.audio_processor.chunks)
                    
                    # Transcribe audio
                    transcript = transcribe_audio(audio_data)
                    if transcript:
                        st.session_state.messages.append({
                            "role": "user",
                            "content": transcript
                        })
                        
                        # Get AI response
                        response = st.session_state.agent.invoke({
                            "input": transcript
                        })
                        
                        # Convert response to speech
                        speech_file = text_to_speech(response['output'])
                        if speech_file:
                            st.audio(speech_file, format="audio/mp3")
                            os.unlink(speech_file)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response['output']
                        })
                    
                    # Clear recording
                    webrtc_ctx.audio_processor.chunks = []
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat history column
    with col2:
        st.markdown('<div class="luxury-container">', unsafe_allow_html=True)
        st.markdown(
            '<h3 style="color: #DAA520; font-family: \'Cinzel\', serif; text-align: center;">'
            '💬 Conversation History</h3>', 
            unsafe_allow_html=True
        )
        
        if not st.session_state.messages:
            st.markdown(
                '<div style="text-align: center; color: #B8860B; font-style: italic;">'
                'Your conversation will appear here...</div>',
                unsafe_allow_html=True
            )
        
        for msg in st.session_state.messages:
            icon = "👤" if msg["role"] == "user" else "👑"
            st.markdown(
                f'<div class="chat-message">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div style="text-align: center; padding: 2rem; color: #B8860B; '
        'font-family: \'Cormorant Garamond\', serif;">'
        'LeChateau - Where Elegance Meets Innovation<br>'
        '<span style="font-size: 0.8rem;">Powered by Advanced AI Technology</span>'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
